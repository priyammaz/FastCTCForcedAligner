import os
import json
import math
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from accelerate import Accelerator
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCTC, AutoProcessor

### Import our aligner ###
from ._aligner import align_batch

### Limit Printing ###
import warnings
import logging
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore")

hf_logging.set_verbosity_error()
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

### Set Defaults ###
SAMPLING_RATE = 16000 # default sampling rate for Wav2Vec2
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def default(input, default):
    return input if input is not None else default

def _check_in_csv(df, col):
    return col in df.columns

def get_durations(paths, num_workers):
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        durations = list(pool.map(lambda p: librosa.get_duration(path=p), paths))
    return durations

def shard(manifest, world_size):
    """Split a list into world_size roughly equal chunks"""
    size = len(manifest) // world_size
    return [manifest[i * size:(i + 1) * size] for i in range(world_size - 1)] + [manifest[(world_size - 1) * size:]]    

def shard_by_duration(manifest, world_size):
    """
    Distribute (path, transcript, duration) tuples across GPUs so each GPU
    gets roughly equal total audio duration, while keeping each GPU's local
    list sorted longest→shortest to minimise padding within batches.

    interleaved round-robin on a globally sorted list:
        Sort all items longest to shortest, then deal like a deck of cards:
            GPU 0 gets indices 0, W, 2W,  ...
            GPU 1 gets indices 1, W+1, 2W+1, ...
            ...
        This spreads duration evenly across ranks. Each rank then re-sorts
        its own slice so its DataLoader batches stay tight.
    """
    sorted_manifest = sorted(manifest, key=lambda x: x[2], reverse=True)

    shards = [[] for _ in range(world_size)]
    for i, item in enumerate(sorted_manifest):
        shards[i % world_size].append(item)

    # Each shard is already nearly sorted due to the interleave, but an
    # explicit sort guarantees it — cost is negligible vs. inference time.
    for shard in shards:
        shard.sort(key=lambda x: x[2], reverse=True)

    return shards

class AudioDataset(Dataset):
    """
    Basic dataset to load audio, resample if necessary, and return
    audio, transcripts and path to audio
    """
    def __init__(self, paths, transcripts, processor):
        self.paths = paths
        self.transcripts = transcripts
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)
        values = self.processor(waveform.squeeze(0), sampling_rate=SAMPLING_RATE).input_values[0]
        return values, self.transcripts[idx], path

def collate_function(processor):
    """
    Collate batch of audios together using the Processor
    """
    def _collate_fn(batch):
        waveforms   = [{"input_values": b[0]} for b in batch]
        transcripts = [b[1] for b in batch]
        paths       = [b[2] for b in batch]
        inputs = processor.pad(
            waveforms,
            padding=True,
            return_attention_mask=True,
            pad_to_multiple_of=SAMPLING_RATE,
            return_tensors="pt",
        )
        return inputs, transcripts, paths
    return _collate_fn

class CTCEmissionExtractor:
    """
    Inference script to get emissions as log probabilities
    """
    def __init__(self, model, accelerator):
        self.model = model
        self.accelerator = accelerator

    @torch.inference_mode()
    def __call__(self, inputs):

        ### Inference Model ###
        inputs = {k: v.to(self.accelerator.device, non_blocking=True) for k, v in inputs.items()}        
        logits = self.model(**inputs).logits

        ### Get True (unpadded) audio lengths ###
        audio_lengths = inputs["attention_mask"].sum(dim=-1)

        ### Get true encoded lengths ###
        encoded_lengths = self.accelerator.unwrap_model(self.model)._get_feat_extract_output_lengths(audio_lengths)
        
        ### Get the number of audio samples each embedding represents (dowsample factors ~320) ###
        samples_per_embed = math.floor(((audio_lengths / encoded_lengths))[0])

        ### Get the actual time (seconds) represented by each embedding ###
        time_per_embed = samples_per_embed / SAMPLING_RATE

        ### Extract out the emissions upto the valid positions ###
        emissions = []
        for i in range(len(encoded_lengths)):
            valid_len = encoded_lengths[i].item()
            valid_logits = logits[i, :valid_len]

            ### Comupte LogProbs on device ###
            log_probs = torch.log_softmax(valid_logits, dim=-1)
        
            ### Transfer to CPU ###
            emissions.append(log_probs.to("cpu", non_blocking=True))

        ### Dont continue until all non_blocking ops are done ###
        if self.accelerator.device.type == "cuda":
            torch.cuda.synchronize()

        return emissions, time_per_embed

class AudioTextAlignment:
    """
    Wrapper on our aligner method with included normalize_fn
    to prepare raw transcripts for the CTC tokenizer
    """
    def __init__(self, processor, blank_token, normalize_fn=None):
        self.processor = processor
        self.char_to_id = self.processor.tokenizer.get_vocab()
        self.blank_token = blank_token
        self.normalize_fn = normalize_fn
        self.token_dictionary = {i: c for i, c in self.char_to_id.items()}

    def process_alignments(
        self,
        batch_emissions,
        transcripts,
    ):
        
        assert len(batch_emissions) == len(transcripts)
   
        return align_batch(batch_emissions, transcripts, 
                           self.token_dictionary, self.blank_token, 
                           normalize_fn=self.normalize_fn,
                           fast=True)

    def __call__(self, batch_emissions, transcripts):
        return self.process_alignments(batch_emissions, transcripts)

def save(alignment, path_to_audio, transcript, duration, word_alignment=None, root=None):

    """
    Quick save method. Creates a json file as :

        {
            "audio_path": path_to_audio,
            "alignment_path": path_to_alignments.npy,
            "duration_seconds": seconds,
            "transcript": "Transcription of the audio", 
            "word_alignments": [
                                    {"word": word, "start": start_time, "end": end_time},
                                    {"word": word, "start": start_time, "end": end_time},
                                    {"word": word, "start": start_time, "end": end_time},
                                    ...
                                ]
        }

    """

    audio_dir = os.path.dirname(path_to_audio)
    audio_stem = os.path.splitext(os.path.basename(path_to_audio))[0]

    if root is None:
        base_path = os.path.join(audio_dir, audio_stem)
    else:
        os.makedirs(root, exist_ok=True)
        base_path = os.path.join(root, audio_stem)

    alignment_path = base_path + "_alignment.npy"
    np.save(alignment_path, np.array(alignment))

    metadata = {
        "audio_path": os.path.abspath(path_to_audio),
        "alignment_path": os.path.abspath(alignment_path),
        "duration_seconds": duration,
        "transcript": transcript,
    }

    if word_alignment is not None:
        metadata["word_alignment"] = word_alignment

    metadata_path = base_path + "_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path

def get_word_alignments(
    char_alignments,
    transcript,
    time_per_embed,
    audio_duration
):
    token_str = '|'.join(transcript.split()).upper()

    assert len(char_alignments) == len(token_str), (
        f"Alignment length {len(char_alignments)} != token string length {len(token_str)}"
    )

    words = []
    current_word_chars = []
    current_word_spans = []

    for char, span in zip(token_str, char_alignments):
        if char == '|':
            if current_word_chars:
                words.append({
                    "word":  ''.join(current_word_chars),
                    "start": min(current_word_spans[0][0]*time_per_embed, audio_duration),
                    "end":   min(current_word_spans[-1][1]*time_per_embed, audio_duration),
                })
            current_word_chars = []
            current_word_spans = []
        else:
            current_word_chars.append(char)
            current_word_spans.append(span)

    if current_word_chars:
        words.append({
            "word":  ''.join(current_word_chars),
            "start": min(current_word_spans[0][0]*time_per_embed, audio_duration),
            "end":   min(current_word_spans[-1][1]*time_per_embed, audio_duration),
        })

    return words

class BulkAligner:

    """
    Force alignment of transcripts to audio distributed onto multiple GPUs with
    alignment computation accelerated with our C++ implementation!

    Args:
        path_to_manifest: Path to csv file that contains paths to audio, transcripts and optionally durations
        ctc_backbone: Huggingface Wav2Vec2 Backbone to use, Default -> facebook/wav2vec2-base-960h
        batch_size: Num samples per GPU, Default -> 16
        blank_token: CTC Blank Token, Default -> "|"
        normalize_fn: Any function postprocessing to apply to raw transcripts, Default -> None
        path_to_audio_column: Name of audio column in manifest, Default -> "path_to_audio"
        transcript_column: Name of transcript column in manifest, Default -> "transcript"
        duration_column: Name of (optional) duration column in manifest, Default -> "duration"
        sort_by_longest: Sort batches to process longest to shrotest to avoid excess wasted padding operations
        num_workers: How many workers for different operations?
        compile_model: Do you want to use torch.compile to speed up inference?
        save_dir: Where do you want to save outputs, if None, will be saved alongside audio files
    """
    def __init__(self, 
                 path_to_manifest,
                 return_word_alignments=False,
                 ctc_backbone="facebook/wav2vec2-base-960h",
                 batch_size=16,
                 blank_token="|",
                 normalize_fn=None,
                 path_to_audio_column=None, 
                 transcript_column=None,
                 duration_column=None,
                 sort_by_longest=True,
                 num_workers=8,
                 compile_model=False,
                 save_dir=None
            ):
        
        ### Do you want word alignments? ###
        self.return_word_alignments = return_word_alignments
        
        ### Where to save alignments ###
        self.save_dir = save_dir

        ### Text processor ###
        self.normalize_fn = normalize_fn

        ### Instantiate Accelerator ###
        self.accelerator = Accelerator(mixed_precision="fp16" if DTYPE == torch.float16 else "bf16", split_batches=False)
        self.world_size = self.accelerator.num_processes
        self.rank = self.accelerator.process_index

        ### Load Model, Processor ###
        self.model = AutoModelForCTC.from_pretrained(
            ctc_backbone, 
            attn_implementation="sdpa",
            torch_dtype=DTYPE
        )
        self.model.eval()

        if compile_model:
            self.model = torch.compile(self.model)

        self.model = self.accelerator.prepare(self.model)
        self.processor = AutoProcessor.from_pretrained(ctc_backbone)
        
        ### Load Emissions Extractor ###
        self.emissions_extractor = CTCEmissionExtractor(self.model, self.accelerator)
        
        ### Load Aligner ###
        self.aligner = AudioTextAlignment(self.processor, blank_token, normalize_fn)

        ### Load Manifest ###
        self.manifest = pd.read_csv(path_to_manifest)

        ### Setup columns and check existence ###
        self.paths_col = default(path_to_audio_column, "path_to_audio")
        self.transcripts_col = default(transcript_column, "transcript")
        self.duration_column = default(duration_column, "duration")
        
        assert _check_in_csv(self.manifest, self.paths_col), f"{self.paths_col} not found in {self.manifest.columns}"
        assert _check_in_csv(self.manifest, self.transcripts_col), f"{self.transcripts_col} not found in {self.manifest.columns}"
        self.compute_durations = not _check_in_csv(self.manifest, self.duration_column)

        ### Grab Paths, Transcripts and Durations (optionally) ###
        self.all_paths = self.manifest[self.paths_col].tolist()
        self.all_transcripts = self.manifest[self.transcripts_col].tolist()

        if sort_by_longest:
            if not self.compute_durations:
                self.durations = self.manifest[self.duration_column].tolist()
            else:
                self.accelerator.print("Computing Audio Durations for Corpus..")
                self.durations = get_durations(self.all_paths, num_workers=num_workers)
            full_manifest = list(zip(self.all_paths, self.all_transcripts, self.durations))
            shards = shard_by_duration(full_manifest, self.world_size)

        else:
            full_manifest = list(zip(self.all_paths, self.all_transcripts))
            shards = shard(full_manifest, self.world_size)
        
        ### Get shard for this worker ###
        rank_manifest = shards[self.rank]
        rank_paths, rank_transcripts, *_ = zip(*rank_manifest)

        ### Prepare DataLoader ###
        total_batches = (len(rank_manifest) + batch_size - 1) // batch_size
        self.rank_pbar = tqdm(
            total=total_batches,
            desc=f"GPU {self.rank}",
            position=self.rank,
            leave=True,
            dynamic_ncols=True,
        )

        dataset = AudioDataset(rank_paths, rank_transcripts, self.processor)
        self.loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            shuffle=False, 
            collate_fn=collate_function(self.processor)
        )        

        self.loader = self.accelerator.prepare(self.loader)

    def align(self):

        for inputs, transcripts, paths in self.loader:
            emissions, time_per_embed  = self.emissions_extractor(inputs)
            alignments = self.aligner(emissions, transcripts)
            for a, p, t in zip(alignments, paths, transcripts):
                d = float(librosa.get_duration(path=p))
                wa = None
                if self.return_word_alignments:
                    t = t.replace("\n", "").strip()
                    t = self.normalize_fn(t)
                    wa = get_word_alignments(a,t,time_per_embed,d)
                save(a,p,t,d,wa,self.save_dir)

            self.rank_pbar.update(1)

        self.accelerator.wait_for_everyone()
        self.rank_pbar.close()
        self.accelerator.end_training()





