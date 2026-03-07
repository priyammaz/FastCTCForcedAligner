"""
Quick example of how to process a bulk of audio/transcript pairs

To use this you can use:

    accelerate launch bulk_process.py to distribute between GPUs or 
    just standard python bulk_process.py for single gpu. 
"""
import re
from ctc_forced_aligner import BulkAligner

def prepare_text(transcript):
    """Wav2Vec2 only has uppercase letters and no punctuation!"""
    return re.sub(r'[^A-Za-z ]', '', transcript).upper()
    
aligner = BulkAligner(
    path_to_manifest="manifest.csv",
    normalize_fn=prepare_text,
    path_to_audio_column="path_to_audio",
    transcript_column="transcript",
    sort_by_longest=True,
    compile_model=True, 
    save_dir="output/",
    return_word_alignments=True,
    num_workers=24,
    batch_size=16
)

aligner.align()

