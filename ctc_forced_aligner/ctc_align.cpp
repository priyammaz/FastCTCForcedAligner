#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

// metadata for each point
struct Point {
    int token_index; // which token 
    int time_index; // which frame
    float score; // what was the prob
};

// metadata for each segment
struct Segment {
    int start; 
    int end;
    float score;
};
    
// emission: (num_frames, vocab_size) row-major numpy float32 array already log softmaxed over vocab size
// tokens: token ids for the transcript
// blank_id: CTC blank token id
std::vector<std::pair<int,int>> align_single(
    py::array_t<float, py::array::c_style | py::array::forcecast> emission_arr,
    const std::vector<int>& tokens,
    int blank_id
) {

    auto buf = emission_arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("emission must be 2D (num_frames, vocab_size)");

    // get the shape and constants
    const int num_frame = static_cast<int>(buf.shape[0]);
    const int vocab_size = static_cast<int>(buf.shape[1]);
    const int num_tokens = static_cast<int>(tokens.size());
    const float* emission = static_cast<float*>(buf.ptr);

    // Build Empty Trellis for each frame by each token flattened to a vector
    // and we populate them with zeros to begin with
    // The trellis starts as 
    //          H      |      I
    // t=0  [  0.0    0.0    0.0  ]
    // t=1  [  0.0    0.0    0.0  ]
    // t=2  [  0.0    0.0    0.0  ]
    // t=3  [  0.0    0.0    0.0  ]
    // t=4  [  0.0    0.0    0.0  ]
    // t=5  [  0.0    0.0    0.0  ]
    std::vector<float> trellis(num_frame * num_tokens, 0.0f);

    // Create a constant for NEG_INF for use later 
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    const float INF = std::numeric_limits<float>::infinity();

    // pointer arithmetic that basically says:
    // emit = lambda t, v: emission[t * vocab_size + v]
    // where t,v is which row,column
    // [&] is the capture clause of what variables can it access
    // from outside the lambda, [&] is just capture everything
    // we need these values so it returns the value
    auto emit = [&](int t, int v) -> float {
        return emission[t * vocab_size + v];
    };

    // same pointer arithmetic from before,
    // but to access the trellis rather than the emissions
    // we want to update this so we return a reference float&
    // and replace with values later!
    auto T = [&](int t, int j) -> float& {
        return trellis[t * num_tokens + j];
    };

    // create variable to store cumulative summations in
    float cumsum = 0.0f;

    // The first column represents, we are still on the first token at 
    // frame T. The only way to stay at H for example the entire time
    // is if we only kept predicting blank tokens. So get the prob of 
    // blank at every timestep from t=1 to the end, and cumulative sum
    // them together!

    //          H      |      I
    // t=0  [  0.0    0.0    0.0  ]
    // t=1  [  0.0    0.0    0.0  ]
    // t=2  [  0.0    0.0    0.0  ]
    // t=3  [  0.0    0.0    0.0  ]
    // t=4  [  0.0    0.0    0.0  ]
    // t=5  [  0.0    0.0    0.0  ]

    //  
    // Say blank log-probs at each frame are `[-0.5, -0.3, -0.8, -0.2, -0.6, -0.4]`.
    // 
    // t=0: cumsum = 0.0 (start, untouched)
    // t=1: cumsum = 0.0 + (-0.5) = -0.5
    // t=2: cumsum = -0.5 + (-0.3) = -0.8
    // t=3: cumsum = -0.8 + (-0.8) = -1.6
    // t=4: cumsum = -1.6 + (-0.2) = -1.8
    // t=5: cumsum = -1.8 + (-0.6) = -2.4
    // 

    // This says: "the score of still being on token H at frame t is the sum of all blank emissions up to t" — the only way to stay on the first token is to keep emitting blanks.
    // 
    //         H       |      I
    // t=0  [  0.0    0.0    0.0  ]
    // t=1  [ -0.5    0.0    0.0  ]
    // t=2  [ -0.8    0.0    0.0  ]
    // t=3  [ -1.6    0.0    0.0  ]
    // t=4  [ -1.8    0.0    0.0  ]
    // t=5  [ -2.4    0.0    0.0  ]

    for (int t = 1; t < num_frame; ++t) {
        cumsum += emit(t, blank_id); // accumualate cumulative sum
        T(t, 0) = cumsum; // set in trellis
    }

    // It is impossible to be past the first token at frame 0. 
    // so we set these to negative infinity!
    //         H       |      I
    // t=0  [  0.0   -inf   -inf  ]  <= can only start at H not blank or I
    // t=1  [ -0.5    0.0    0.0  ]
    // t=2  [ -0.8    0.0    0.0  ]
    // t=3  [ -1.6    0.0    0.0  ]
    // t=4  [ -1.8    0.0    0.0  ]
    // t=5  [ -2.4    0.0    0.0  ]
    for (int j = 1; j < num_tokens; ++j)
        T(0, j) = NEG_INF;
        
    // In the example we have 6 frames and 3 tokens. The backtrack will start
    // at the bottom right corner T(5,2) and trace its way back to the top-left. 
    // But imagine a scenario like this:
    //         H     |      I
    // t=0  [  x   -inf   -inf  ]  
    // t=1  [  x    0.0    0.0  ]
    // t=2  [  x    0.0    0.0  ]
    // t=3  [  x    0.0    0.0  ]
    // t=4  [  x    0.0    0.0  ]
    // t=5  [  x    0.0    0.0  ]
    
    // where the x indicates that we selected H every time (via blank tokens)
    // then by the time we get to the end t=5, we have no way to fit the rest
    // of our tokens | and H!. If our blank emmisions have a high log prob, 
    // our dynamic programming may prefer to stay on H all the way to frame 5
    // and this would break as there are no frames left to assign the remaining
    // tokens. Thus we use the +inf trick!
    // Because we have a total number of tokens 3, and we start at the first one
    // if we stay at H the entire time, we need atleast 2 timesteps to make it to
    // the end! So lets force it to happen:
    //         H       |      I
    // t=0  [  0.0   -inf   -inf  ]
    // t=1  [ -0.5    0.0    0.0  ]
    // t=2  [ -0.8    0.0    0.0  ]
    // t=3  [ -1.6    0.0    0.0  ]
    // t=4  [ +inf    0.0    0.0  ]
    // t=5  [ +inf    0.0    0.0  ]
    
    // now during backtracking we check. At t=5, j=1 (on token |)
    // we compare the probabilities:
    //  stay: T(4,1) + p_stay -> from 5 to 4 stay at token 1 = some prob
    //  changed: T(4,0) + p_change -> from 5 to 4 transition from token 1 to token 0 = inf
    // in this case changed will always be inf and greater, thus forcing a change to token 0!

    //         H       |       I
    // t=0  [  y       x       x   ]
    // t=1  [  y       y       x   ]
    // t=2  [  y       y       y   ]
    // t=3  [  y       y       y   ]  <= last valid frame for H
    // t=4  [  x       y       y   ]  <= force transition (+inf)
    // t=5  [  x       y       y   ]  <= force transition (+inf)

    // This makes sure that we force atleast 2 transitions in this case,
    // so even if we stay at H the entire time we are ok!
    for (int t = num_frame - num_tokens + 1; t < num_frame; ++t)
        T(t, 0) = INF;

    // Now we compute the accumulated transition probabilities!
    // for every frame and for every token (after the first one
    // as the first token H has already been handled with initialization 
    // measure the probability of staying at a position vs changing
    // we accumulate this up over first all the tokens, and then move to the
    // next timestep to repeat this again!

    // create a vector to prefetch emissions for time t
    std::vector<float> tok_emit(num_tokens);
    for (int t = 0; t < num_frame - 1; ++t) {
        // get the blank id prob for this frame
        const float blank_t = emit(t, blank_id);
        
        // prefetch 
        for (int j = 1; j < num_tokens; ++j) {
            tok_emit[j] = emit(t, tokens[j]);
        }
           
        // perform op
        for (int j = 1; j < num_tokens; ++j) {
            // to stay at some j we need a blank token
            // so T(t,j) is my cumulative prob until now and we 
            // add the prob of getting another blank token 
            float stay = T(t, j) + blank_t;

            // or we can transition from some j-1 to j
            // this is what is the cumulative prob from the 
            // earlier character to the one I am on now
            float change = T(t, j - 1) + tok_emit[j];

            // The viterbi algorithm only keeps the best (most likely)
            // path rather than the sum of all paths
            T(t + 1, j) = std::max(stay, change);
        }
    }

    // Backtracking through the compute trellis 
    // now that we have all our cumulative log probs
    // to all the time/token positions in the trellis, 
    // we start on the bottom right (the last frame and last token)
    // and work our way back to find the overall most likely path
    // for example we have here an example final trellis, 
    // we will start at the bottom right
    //          H      |       I
    // t=0  [  0.0   -inf    -inf  ]
    // t=1  [ -0.5   -1.2    -inf  ]
    // t=2  [ -0.8   -0.9    -2.7  ]
    // t=3  [ +inf   -1.4    -3.1  ]  <= start here (t=3, j=2)

    // first we create an empty vector called path full of Point objects
    std::vector<Point> path;

    // ensure the vector has num_frames positions (an upper bound) preallocated
    path.reserve(num_frame);

    // our starting point is the bottom right lets grab the indexes for that
    int t = num_frame - 1;
    int j = num_tokens - 1;
    
    // append to the path this point at the starting bottom right
    path.push_back(Point{j, t, std::exp(emit(t, blank_id))});

    while (j > 0) { // just keep going until we are back to token 0

        // make sure we have enough timesteps, or we ran out! this 
        // should be handled by our +inf from earlier but just incase!
        if (t <= 0) throw std::runtime_error("backtrack failed: t <= 0");
        
        // lookup the emission probs of the previous frame so we can determine how
        // we arrived to this position in the trellis. We basically look at the 
        // previous timestep and ask what the model had predicted there:

        // what was the log prob the model assigned to a blank token at t-1. If we 
        // stayed on the same token j, we must have emitted a blank
        float p_stay = emit(t-1, blank_id); 

        // what is the log prob of token j at frame t-1. If we advanced from j-1 to j, we must have
        // emmited token j
        float p_change = emit(t-1, tokens[j]); 
        
        // what was the probability we stayed? That is the probability up to the previous
        // timestep at token j, followed by the blank token probability
        // this is the same operation as the forward pass!
        float stayed = T(t - 1, j) + p_stay; 

        // what was the probability we transitioned? That is the probability upto the previous
        // timestep at token j-1 followed by the probability of current token j
        // this is the same operation as the forward pass!
        float changed = T(t - 1, j - 1) + p_change;

        // go back a frame
        t -= 1;
        
        // if we changed, then reverse a step here as well
        if (changed > stayed) j -= 1;

        // get the actual probability? we have logprobs, exponentiate to get values
        // p_change if changed > stayed else p_stay
        float prob = std::exp(changed > stayed ? p_change : p_stay);

        // append this on!
        path.push_back({j, t, prob});
    }

    // After backtracking is done here is an example of what it looks like!
    //          H      |       I
    // t=0  [  0.0   -inf    -inf  ]
    // t=1  [ -0.5   -inf    -inf  ] 
    // t=2  [ -0.8   -0.9    -inf  ]  
    // t=3  [ +inf   -1.4    -3.1  ]

    // and we have essentially created a list like:
    // path = [
    //     Point(j=2, t=3),   
    //     Point(j=1, t=3),
    //     Point(j=1, t=2),
    //     Point(j=0, t=2),
    // ]

    // But notice that if H starts at Point(j=0, t=0), and our backtrack
    // ended at Point(j=0, t=2), what happend to Point(j=0, t=1)? We 
    // missed this, so we must manually add them in! Essentially, any 
    // extra frames we have left over in the j=0 column, we must just
    // manually add them in! and we already did our cumulative prod
    // until now so its just a matter of appending

    while (t > 0) { // t is not 0 yet in this case, it is at 2
        float prob = std::exp(emit(t - 1, blank_id));
        path.push_back({j, t - 1, prob});
        t -= 1;
    }

    // and we have finally gotten
    // path = [
    //     Point(j=2, t=3),   
    //     Point(j=1, t=3),
    //     Point(j=1, t=2),
    //     Point(j=0, t=2),
    //     Point(j=0, t=1),
    //     Point(j=0, t=0),
    // ]

    // we want our path from start to end so reverse!
    std::reverse(path.begin(), path.end());

    // some tokens are repeated, like our | token is used twice 
    // both at t=3 and t=2. Lets go ahead and just merge repeated
    // things together!

    // create an empty list to store segment objects
    std::vector<Segment> segments;

    // i1 stores the start of a run, i2 will scan forward until the token changes
    int i1 = 0;
    while (i1 < (int)path.size()) { // for all positions from the start to end of the list of paths
        int i2 = i1; // i2 shoudl start checking from the starting point of i1

        // as long as i2 is inside the bounds of the number of elements 
        // and our token indexes are the same (being repeated) keep going
        while (i2 < (int)path.size() && path[i1].token_index == path[i2].token_index)
            i2++;

        // i2 will stop at the end, so now lets accumulate the total score 
        float score = 0.0f;

        // for all elements from i1 to i2, get average prob
        for (int k = i1; k < i2; ++k) score += path[k].score;
        score /= (i2 - i1);

        // append to segments
        // i2-1 is one past the end of the run , i2-1 makes it the last point in the 
        // run, and the +1 makes it an exlusive end. So in python slicing
        // [0:5] is [0,1,2,3,4], we will store 0 to 5 in our case
        segments.push_back({path[i1].time_index, path[i2-1].time_index + 1, score});
        
        // push i2 up so while loop can catch it when we are done
        i1 = i2;
    }

    // Our return is a vector, where each element is a tuple of ints (starting/ending point)
    std::vector<std::pair<int,int>> result;

    // preallocate memory for the vector
    result.reserve(segments.size());

    // loop through each segment, this is a range based loop
    // where each element of segments is assigned to variable seg. 
    // but instead of creating a variable copy we just get the 
    // address to seg and autoset type
    for (auto& seg : segments)

        // emplace_back is a little more efficient as it will
        // directly construct the object in the containers memory, rather 
        // than push_back that has a temporary/delete operation
        result.emplace_back(seg.start, seg.end);

    return result;
}

std::vector<std::vector<std::pair<int,int>>> align_batch(
    const std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>>& emissions,
    const std::vector<std::vector<int>>& token_seqs,
    int blank_id
) {

    if (emissions.size() != token_seqs.size())
        throw std::runtime_error("emissions and token_seqs must have same length");

    const int n = static_cast<int>(emissions.size()); // batch size
    std::vector<std::vector<std::pair<int,int>>> results(n); // create a vector to store

    // apply to each sample one at a time!
    for (int i = 0; i < n; ++i)
        // write as fine as each operation is indexed to i, and performs
        // the write to position i
        results[i] = align_single(emissions[i], token_seqs[i], blank_id);

    return results;
}

PYBIND11_MODULE(_ctc_align_cpp, m) {
    m.doc() = "Fast CTC forced alignment (trellis + backtrack) in C++";

    m.def("align_single", &align_single,
        py::arg("emission"),
        py::arg("tokens"),
        py::arg("blank_id") = 0,
        R"(
Align a single emission matrix to a token sequence.

Args:
    emission: np.ndarray of shape (num_frames, vocab_size), float32, raw logits (log_softmax applied internally)
    tokens:   list[int] token ids for the transcript
    blank_id: CTC blank token id (default 0)

Returns:
    list of (start_frame, end_frame) tuples, one per token
        )"
    );

    m.def("align_batch", &align_batch,
        py::arg("emissions"),
        py::arg("token_seqs"),
        py::arg("blank_id") = 0,
        R"(
Align a batch of emissions to a batch of token sequences 

Args:
    emissions: list of np.ndarray, each (num_frames, vocab_size), float32
    token_seqs: list of list[int]
    blank_id: CTC blank token id

Returns:
    list of list of (start_frame, end_frame) tuples
        )"
    );
}