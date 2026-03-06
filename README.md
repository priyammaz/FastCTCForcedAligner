# FastCTCForcedAligner

Forced alignment is a key preprocessing step in Speech Research to time align text to cooresponding audio. This repository provides a fast CTC-Based Force aligner with a C++ Backend on multiple GPUs, to enable bulk force alignment. 

CTC Emission Matricies are extracted from Huggingface 🤗 Wav2Vec2 models and the force alignment finds the exact frame level time of every character and word in the audio. This package provides:

- **Python** implementation inspired by the [official PyTorch Tutorial](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html)
- **C++ Extension** for fast single and batch alignments
- **CLI** interface for each useage

## Installation

```bash
git clone https://github.com/priyammaz/FastCTCForcedAligner.git
cd FastCTCForcedAligner

pip install .
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
word_alignments = get_word_alignments(char_alignments transcript, time_per_embed, duration)
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

## Bulk Alignment - MultiGPU Batched Alignment

In some cases we want to force align a large collection of audio/transcript pairs. Our bulk alignment automates this process for you!

### Data Preparation

Audios are provided through a simple CSV file of path/transcript pairs. 

| path_to_audio | transcript |
| :--- | :--- |
| path_to_audio/audio1.mp3 | hello world |
| path_to_audio/audio2.mp3 | lets get aligned! |
...

Additionally you can provide some extra metadata in the CSV:

1. ```duration```: if durations are provided, then during sorting from longest to shortest (if enabled for efficiency), they will not have to be computed and will read from this column

2. ```id```: By default, if the name of an audio file is *audio1.mp3*, then all cooresponding metadata will be saved in the output file as *audio1.npy*, *audio1.json*, ... If you want to name the generated metadata files differently from the original audio files, you can provide an **id** column and that name will be used instead for each file. 


### Distributed Setup

All distributed inference setup is handled through Huggingface 🤗 Accelerate! This means you need to setup accelerate yourself to your specific machine. Luckily this is easy!

```bash
accelerate config # follow the questions asked
```

### CLI

You can use the following command line method to initiate bulk inference (defaults are for the Wav2Vec2 Model!)

```bash
bulk-align --manifest "path_to_manifest.csv" \         # csv with metadata
           --save-dir "output/" \                      # output directory
           --backbone "facebook/wav2vec2-base-960h" \  # What CTC backbone do you want?
           --compile \                                 # torch compile for faster inference
           --batch_size 16 \                           # batch inference 
           --num_workers 8 \                           # how many cpu workers?
```

### Python

If you would rather write a python script you can do the following:

```python
~run.py~

import re
from ctc_forced_aligner import BulkAligner

def process_text(text):
    """Default normalization for Wav2Vec2 — uppercase letters and spaces only."""
    return re.sub(r"[^A-Za-z ]", "", text).upper()

aligner = BulkAligner(
    path_to_manifest="manifest.csv",            # csv with metadata
    ctc_backbone="facebook/wav2vec2-base-960h", # What CTC backbone do you want?
    batch_size=16,                              # batch inference 
    normalize_fn=process_text,                  # text process function
    sort_by_longest=True,                       # efficient by reducing padding (default True in cli)
    num_workers=8,                              # how many cpu workers?
    compile_model=True,                         # torch compile for faster inference
    save_dir="output/",                         # output directory
    return_word_alignments=True                 # Do you want word level alignments (default True in cli)
)

aligner.align()

```

When running this script make sure to use ```accelerate launch run.py``` to trigger distributed inference!

### Output

When running this two files will be stored in your selected output directory:

#### Alignments

`*_alignments.npy` contains a NumPy array of shape `(transcript_length, 2)`. Each row corresponds to a character in the transcript and stores the **start** and **end (exclusive)** indices of the aligned region in the Wav2Vec2 embedding sequence.

For Example
Transcript:

CAT 

Alignment array:

```python
array([
    [0, 3],   # 'C'
    [3, 7],   # 'A'
    [7, 10]   # 'T'
])
```

This means our forced alignment says embedding indices 0–2 align to "C", indices 3–6 align to "A", and indices 7–9 align to "T".

#### Metadata

For each audio file we also produce the metadata file ```*_metadata.json``` that provides us the path to the original audio, the path to the computed alignments, the duration of the audio file, the transcript, as well as the optionally computed word alignments

```json
{
    "audio_path": path_to_audio,
    "alignment_path": path_to_alignments.npy,
    "duration_seconds": seconds,
    "transcript": "Transcription of the audio", 
    "word_alignments": [
        {"word": word1, "start": start_time, "end": end_time},
        {"word": word2, "start": start_time, "end": end_time},
        {"word": word3, "start": start_time, "end": end_time},
        ...
    ]
}
```