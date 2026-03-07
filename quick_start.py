import re
import librosa
from transformers import AutoModelForCTC, AutoProcessor
from ctc_forced_aligner import align, load_for_inference, \
    get_word_alignments, CTCEmissionExtractor

path_to_audio = "LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"
transcript = "CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RAaaaCHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK"
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
char_alignments = align(emissions, transcript, char2id, blank_token="|", fast=True)
# -> [(start_idx, end_idx), (start_idx, end_idx), ...]

### Compute Word Alignments ###
duration = librosa.get_duration(path=path_to_audio)
word_alignments = get_word_alignments(char_alignments, transcript, 
                                      time_per_embed, duration)

# -> [{"word": word, "start": start_time, "end": "end_time"},  ...]