import os
import json
import math
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
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
hf_logging.disable_progress_bar() 
hf_logging.disable_default_handler()
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

### Set Defaults ###
SAMPLING_RATE = 16000 # default sampling rate for Wav2Vec2
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def default(input, default):
    return input if input is not None else default

def _check_in_csv(df, col):
    return col in df.columns

def get_durations(paths, num_workers, show_progress=True):
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        durations = list(tqdm(
            pool.map(lambda p: librosa.get_duration(path=p), paths),
            total=len(paths),
            desc="Extracting Audio Durations",
            colour="cyan",
            disable=not show_progress,
        ))
    return durations

def _fmt_duration(seconds):
    """Format seconds as Xh Xm Xs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.0f}s"
    elif m > 0:
        return f"{m}m {s:.0f}s"
    return f"{s:.1f}s"

def _print_banner(config: dict):
    """Print a formatted info banner. Only called from rank 0."""
    W = 60
    print()
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + "  CTC Forced Aligner".center(W - 2) + "│")
    print("├" + "─" * (W - 2) + "┤")
    for key, val in config.items():
        line = f"  {key:<22}{val}"
        print("│" + f"{line:<{W-2}}" + "│")
    print("└" + "─" * (W - 2) + "┘")
    print()


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

def load_for_inference(path_to_audio, 
                       processor):
    """
    Data prep for single input inference
    """
    waveform, sr = torchaudio.load(path_to_audio)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != SAMPLING_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)
    values = processor(waveform.squeeze(0), sampling_rate=SAMPLING_RATE).input_values[0]
    values = torch.tensor(values).unsqueeze(0)
    return {"input_values": values}

class AudioDataset(Dataset):
    """
    Basic dataset to load audio, resample if necessary, and return
    audio, transcripts and path to audio
    """
    def __init__(self, paths, transcripts, ids, processor):
        self.paths = paths
        self.transcripts = transcripts
        self.ids = ids
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        values = load_for_inference(path, self.processor)["input_values"].squeeze(0)
        return values, self.transcripts[idx], path, self.ids[idx]

def collate_function(processor):
    """
    Collate batch of audios together using the Processor
    """
    def _collate_fn(batch):
        waveforms = [{"input_values": b[0]} for b in batch]
        transcripts = [b[1] for b in batch]
        paths = [b[2] for b in batch]
        ids = [b[3] for b in batch]

        inputs = processor.pad(
            waveforms,
            padding=True,
            return_attention_mask=True,
            pad_to_multiple_of=SAMPLING_RATE,
            return_tensors="pt",
        )
        return inputs, transcripts, paths, ids
    return _collate_fn

class CTCEmissionExtractor:
    """
    Inference script to get emissions as log probabilities
    """
    def __init__(self, model, accelerator=None):
        self.model = model
        self.accelerator = accelerator
        self.device = str(self.accelerator.device) if accelerator else (str(next(model.parameters()).device))

    @torch.inference_mode()
    def __call__(self, inputs):

        ### Inference Model ###
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}        
        logits = self.model(**inputs).logits

        ### Get True (unpadded) audio lengths ###
        if "attention_mask" in inputs.keys():
            audio_lengths = inputs["attention_mask"].sum(dim=-1)
        else:
            audio_lengths = torch.tensor([inputs["input_values"].shape[-1]])

        ### Get true encoded lengths ###
        if self.accelerator is not None:
            encoded_lengths = self.accelerator.unwrap_model(self.model)._get_feat_extract_output_lengths(audio_lengths)
        else:
            encoded_lengths = self.model._get_feat_extract_output_lengths(audio_lengths)
        
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
        if "cuda" in self.device:
            torch.cuda.synchronize()

        ### Unpack for single inference ###
        if len(emissions) == 1:
            emissions = emissions[0]

        return emissions, time_per_embed

class AudioTextAlignment:
    """
    Wrapper on our aligner method with included normalize_fn
    to prepare raw transcripts for the CTC tokenizer
    """
    def __init__(self, processor, blank_token, normalize_fn=None, num_workers=None):
        self.processor = processor
        self.char_to_id = self.processor.tokenizer.get_vocab()
        self.blank_token = blank_token
        self.normalize_fn = normalize_fn
        self.num_workers = num_workers
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
                           fast=True, num_workers=self.num_workers)

    def __call__(self, batch_emissions, transcripts):
        return self.process_alignments(batch_emissions, transcripts)

def save(alignment, 
         path_to_audio, 
         transcript, duration, 
         word_alignment=None, 
         root=None,
         file_id=None):

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
    stem = str(file_id) if file_id is not None else os.path.splitext(os.path.basename(path_to_audio))[0]
    base_path = os.path.join(root if root is not None else audio_dir, stem)

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
    audio_duration,
    blank_token="|",
):
    token_str = blank_token.join(transcript.split()).upper()

    assert len(char_alignments) == len(token_str), (
        f"Alignment length {len(char_alignments)} != token string length {len(token_str)}"
    )

    words = []
    current_word_chars = []
    current_word_spans = []

    for char, span in zip(token_str, char_alignments):
        if char == blank_token:
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
        save_dir: Where do you want to save outputs?
        ctc_backbone: Huggingface Wav2Vec2 Backbone to use, Default -> facebook/wav2vec2-base-960h
        batch_size: Num samples per GPU, Default -> 16
        blank_token: CTC Blank Token, Default -> "|"
        normalize_fn: Any function postprocessing to apply to raw transcripts, Default -> None
        path_to_audio_column: Name of audio column in manifest, Default -> "path_to_audio"
        transcript_column: Name of transcript column in manifest, Default -> "transcript"
        duration_column: Name of (optional) duration column in manifest, Default -> "duration"
        id_column: Name of (optional) name column you want to use as id for audio, Default -> "id"
        sort_by_longest: Sort batches to process longest to shrotest to avoid excess wasted padding operations
        num_workers: How many workers for different operations?
        compile_model: Do you want to use torch.compile to speed up inference?
    """
    def __init__(self, 
                 path_to_manifest,
                 save_dir,
                 return_word_alignments=False,
                 ctc_backbone="facebook/wav2vec2-base-960h",
                 batch_size=16,
                 blank_token="|",
                 normalize_fn=None,
                 path_to_audio_column=None, 
                 transcript_column=None,
                 duration_column=None,
                 id_column=None,
                 sort_by_longest=True,
                 num_workers=8,
                 compile_model=False
            ):
        
        ### Do you want word alignments? ###
        self.return_word_alignments = return_word_alignments
        
        ### Where to save alignments ###
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        ### Text processor ###
        self.normalize_fn = normalize_fn

        ### How mnay workers? ###
        self.num_workers = num_workers

        ### Instantiate Accelerator ###
        dynamo = TorchDynamoPlugin(backend="inductor", dynamic=True)
        self.accelerator = Accelerator(mixed_precision="fp16" if DTYPE == torch.float16 else "bf16", 
                                       split_batches=False,
                                       dynamo_plugin=dynamo if compile_model else None)
        self.world_size = self.accelerator.num_processes
        self.rank = self.accelerator.process_index

        ### Load Model, Processor ###
        self.model = AutoModelForCTC.from_pretrained(
            ctc_backbone, 
            attn_implementation="sdpa",
            torch_dtype=DTYPE
        )
        self.model.eval()

        self.model = self.accelerator.prepare(self.model)
        self.processor = AutoProcessor.from_pretrained(ctc_backbone)
        
        ### Load Emissions Extractor ###
        self.emissions_extractor = CTCEmissionExtractor(self.model, self.accelerator)
        
        ### Load Aligner ###
        self.aligner = AudioTextAlignment(self.processor, blank_token, normalize_fn, num_workers=num_workers)

        ### Load Manifest ###
        self.manifest = pd.read_csv(path_to_manifest)

        ### Setup columns and check existence ###
        self.paths_col = default(path_to_audio_column, "path_to_audio")
        self.transcripts_col = default(transcript_column, "transcript")
        self.duration_column = default(duration_column, "duration")
        self.id_column = default(id_column, "id")
        
        assert _check_in_csv(self.manifest, self.paths_col), f"{self.paths_col} not found in {self.manifest.columns}"
        assert _check_in_csv(self.manifest, self.transcripts_col), f"{self.transcripts_col} not found in {self.manifest.columns}"
        self.compute_durations = not _check_in_csv(self.manifest, self.duration_column)
        
        ### Grab Paths, Transcripts and Durations (optionally) ###
        self.all_paths = self.manifest[self.paths_col].tolist()
        self.all_transcripts = self.manifest[self.transcripts_col].tolist()
        self.all_ids = self.manifest[self.id_column].tolist() if self.id_column and _check_in_csv(self.manifest, self.id_column) else None

        if sort_by_longest:
            if not self.compute_durations:
                self.durations = self.manifest[self.duration_column].tolist()
            else:
                self.durations = get_durations(self.all_paths, num_workers=num_workers, show_progress=self.accelerator.is_main_process)
            total_hours = sum(self.durations) / 3600
            full_manifest = list(zip(self.all_paths, self.all_transcripts, self.durations, 
                             self.all_ids or [None]*len(self.all_paths)))
            shards = shard_by_duration(full_manifest, self.world_size)

        else:
            full_manifest = list(zip(self.all_paths, self.all_transcripts, 
                             self.all_ids or [None]*len(self.all_paths)))
            shards = shard(full_manifest, self.world_size)
        
        ### Get shard for this worker ###
        rank_manifest = shards[self.rank]
        rank_paths, rank_transcripts, *rest = zip(*rank_manifest)
        rank_ids = rest[-1] if self.all_ids else [None] * len(rank_paths)

        if self.accelerator.is_main_process:
            dtype_str = "bfloat16" if DTYPE == torch.bfloat16 else "float16"
            config = {
                "Model":           ctc_backbone,
                "Dtype":           dtype_str,
                "GPUs":            str(self.world_size),
                "Total files":     f"{len(self.all_paths):,}",
                "Batch size":      str(batch_size),
                "Workers":         str(num_workers),
                "Sort by length":  "yes" if sort_by_longest else "no",
                "Word alignments": "yes" if return_word_alignments else "no",
                "Compile model":   "yes" if compile_model else "no",
                "Save dir":        save_dir or "(alongside audio)",
            }
            if total_hours is not None:
                config["Total audio"] = _fmt_duration(sum(self.durations))
            _print_banner(config)

        ### Prepare DataLoader ###
        total_batches = (len(rank_manifest) + batch_size - 1) // batch_size
        self.rank_pbar = tqdm(
            total=total_batches,
            desc=f"GPU {self.rank}",
            position=self.rank,
            leave=True,
            dynamic_ncols=True,
            colour="green",
            bar_format="{desc} │{bar:35}│ {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]",
        )

        dataset = AudioDataset(rank_paths, rank_transcripts, rank_ids, self.processor)
        self.loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            shuffle=False, 
            collate_fn=collate_function(self.processor)
        )        

    def align(self):
        
        def process_sample(a, p, t, time_per_embed, file_id):
            d  = float(librosa.get_duration(path=p))
            t  = t.replace("\n", "").strip()
            wa = None
            if self.return_word_alignments:
                nt = self.normalize_fn(t) if self.normalize_fn else t
                wa = get_word_alignments(a, nt, time_per_embed, d)
            save(a, p, t, d, wa, self.save_dir, file_id)

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            for inputs, transcripts, paths, ids in self.loader:
                emissions, time_per_embed = self.emissions_extractor(inputs)
                alignments = self.aligner(emissions, transcripts)

                futures = [
                    pool.submit(process_sample, a, p, t, time_per_embed, i)
                    for a, p, t, i in zip(alignments, paths, transcripts, ids)
                ]

                for f in futures:
                    f.result()

                self.rank_pbar.update(1)

        self.rank_pbar.close()
        self.accelerator.end_training()





