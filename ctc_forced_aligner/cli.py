import argparse
import os
import re
import subprocess
import sys

def _normalize(text: str) -> str:
    """Default normalization for Wav2Vec2 — uppercase letters and spaces only."""
    return re.sub(r"[^A-Za-z ]", "", text).upper()

def _is_launched() -> bool:
    """True if we are already inside an accelerate/torch distributed launch."""
    return (
        "RANK" in os.environ
        or "LOCAL_RANK" in os.environ
    )

def _relaunch(num_processes=None) -> None:
    """Re-exec under accelerate launch, optionally overriding num_processes."""
    script = sys.argv[0]

    clean_argv = []
    skip_next  = False
    for tok in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--num-processes", "--num_processes"):
            skip_next = True
            continue
        if tok.startswith(("--num-processes=", "--num_processes=")):
            continue
        clean_argv.append(tok)

    cmd = ["accelerate", "launch"]
    if num_processes is not None:
        cmd += ["--num_processes", str(num_processes)]
    cmd += [script, *clean_argv]

    sys.exit(subprocess.run(cmd).returncode)

def build_parser():
    p = argparse.ArgumentParser(
        prog="ctc-align",
        description=(
            "CTC forced alignment — auto-launches under accelerate using "
            "your existing config. Pass --num-processes to override GPU count."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--manifest", required=True,
                   help="Path to CSV manifest file.")

    # Distributed override
    p.add_argument("--num-processes", type=int, default=None,
                   help="Number of GPUs. If not set, uses accelerate config.")

    # Model
    p.add_argument("--backbone", default="facebook/wav2vec2-base-960h",
                   help="HuggingFace CTC model to use.")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile (inductor backend).")

    # Data columns
    p.add_argument("--audio-column",      default=None,
                   help="Manifest column for audio file paths.")
    p.add_argument("--transcript-column", default=None,
                   help="Manifest column for transcripts.")
    p.add_argument("--duration-column",   default=None,
                   help="Manifest column for pre-computed durations (optional).")

    # Normalization
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable default normalization (uppercase + strip non-alpha). "
                        "Use if transcripts are already clean.")

    # Alignment
    p.add_argument("--blank-token",     default="|",
                   help="CTC blank token string.")
    
    # Batching / workers
    p.add_argument("--batch-size",  type=int, default=16,
                   help="Samples per GPU per batch.")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Worker threads for DataLoader, alignment, and saving.")
    p.add_argument("--no-sort",     action="store_true",
                   help="Disable longest-first duration sorting.")

    # Output
    p.add_argument("--save-dir", default=None,
                   help="Output directory. Defaults to alongside each audio file.")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not _is_launched():
        _relaunch(num_processes=args.num_processes)

    from ctc_forced_aligner import BulkAligner

    normalize_fn = None if args.no_normalize else _normalize

    aligner = BulkAligner(
        path_to_manifest = args.manifest,
        ctc_backbone = args.backbone,
        batch_size = args.batch_size,
        blank_token = args.blank_token,
        normalize_fn = normalize_fn,
        path_to_audio_column = args.audio_column,
        transcript_column = args.transcript_column,
        duration_column = args.duration_column,
        sort_by_longest = not args.no_sort,
        num_workers = args.num_workers,
        compile_model = args.compile,
        save_dir = args.save_dir,
        return_word_alignments=True,
    )

    aligner.align()

if __name__ == "__main__":
    main()


