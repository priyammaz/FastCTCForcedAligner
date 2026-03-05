"""
Code taken from PyTorch Forced Aligner Tutorial
https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
"""

from dataclasses import dataclass
import torch

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

def align_single_py(emission, transcript, token_dictionary, blank_token="|", return_trellis=False):

    if not isinstance(emission, torch.Tensor):
        emission = torch.tensor(emission)
    
    blank_token_id = token_dictionary[blank_token]
    transcript_for_alignment = blank_token.join(transcript.split()).upper()
    tokens = [token_dictionary[c] for c in transcript_for_alignment]
    
    blank_id = blank_token_id
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    
    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")
    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    
    t, j = trellis.size(0) - 1, trellis.size(1) - 1
    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        assert t > 0
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change
        t -= 1
        if changed > stayed:
            j -= 1
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1
    path = path[::-1]
    
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(Segment(
            transcript_for_alignment[path[i1].token_index],
            path[i1].time_index,
            path[i2 - 1].time_index + 1,
            score,
        ))
        i1 = i2

    spans = [(seg.start, seg.end) for seg in segments]
    if not return_trellis:
        return spans
    else:
        return spans, trellis