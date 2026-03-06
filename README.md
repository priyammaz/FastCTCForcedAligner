# FastCTCForcedAligner

Forced alignment is a key preprocessing step in Speech Research to time align text to cooresponding audio. This repository provides a fast CTC-Based Force aligner with a C++ Backend on multiple GPUs, to enable bulk force alignment. 

CTC Emission Matricies are extracted from Huggingface 🤗  Wav2Vec2 models and the force alignment finds the exact frame level time of every character and word in the audio. This package provides:

- **Python** implementation inspired by the [official PyTorch Tutorial](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html)
- **C++ Extension** for fast single and batch alignments
- **CLI** interface for each useage

## Installation

```bash
git clone https://github.com/priyammaz/FastCTCForcedAligner.git
cd FastCTCForcedAligner

pip install -e .
``` 

## Quick Start

```python
import re
import librosa
from transformers import AutoModelForCTC, AutoProcessor
from ctc_forced_aligner import align, load_for_inference, \
    get_word_alignments, CTCEmissionExtractor

path_to_audio = "<path_to_audio>"
transcript = "Hello World, I want to be aligned!"
device = "cuda"

### Set the Huggingface Backbone
ctc_model_backbone = "facebook/wav2vec2-base-960h"
model = AutoModelForCTC.from_pretrained(ctc_model_backbone).to(device)
processor = AutoProcessor.from_pretrained(ctc_model_backbone)

### Load Audio ###
inputs = load_for_inference(path_to_audio, processor)

### Extract Emissions (log probs) ###
extractor = CTCEmissionExtractor(model)
emissions, time_per_embed = extractor(inputs) # -> (num_timesteps x vocab_size)

### Prepare Text ###
def prep_text(text):
    """wav2vec2 has only uppercase and space characters"""
    return re.sub(r"[^A-Za-z ]", "", text).upper()
transcript = prep_text(transcript)

### Compute Character Alignment ###
char2id = processor.tokenizer.get_vocab()
char_alignments = align(emissions, transcript, char2id, fast=True)
# -> [(start_idx, end_idx), (start_idx, end_idx), ...]

### Compute Word Alignments ###
duration = librosa.get_duration(path=path_to_audio)
word_alignments = get_word_alignments(char_alignments, transcript, 
                                      time_per_embed, duration)

# -> [{"word": word, "start": start_time, "end": "end_time"},  ...]
```

## Usage

```align```

Main alignment method to force align a transcript to audio

```python
from ctc_forced_aligner import align

spans = align(
    emission,             # (T, V) tensor or numpy array of log-probs
    transcript,           #  plain-text string
    token_dictionary,     # dict mapping character → token id
    blank_token,          # CTC blank token string, e.g. "|"
    normalize_fn=None,    # optional function to clean the transcript
    fast=True,            # use C++ backend if available
    return_trellis=False, # return trellis (defaults to python backend)
)

# returns: list of (start_frame, end_frame) tuples, one per character
# if return_trellis=True: returns (spans, trellis) — forces Python backend

```

```align_batch```

Align a batch of emissions in parallel 

```python
from ctc_forced_aligner import align_batch

results = align_batch(
    emissions,          # list of (T, V) tensors or arrays
    transcripts,        # list of strings
    token_dictionary,
    blank_token,
    normalize_fn=None,
    fast=True,
    return_trellis=False,
    num_workers=None,   # defaults to os.cpu_count()
)

```

```get_word_alignments```

The previous method ```align``` will return the indexes of character level spans (what indexes of Wav2Vec2 embedding coorespond to each character?). To combine this in to word level spans you can use the following:

```python
from ctc_forced_aligner.bulk_aligner import get_word_alignments

# time_per_embed: seconds per emission frame (from CTCEmissionExtractor)
words = get_word_alignments(spans, transcript, time_per_embed, audio_duration)
```

### Bulk Alignment - MultiGPU Batched Alignment


