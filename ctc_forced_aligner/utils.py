from typing import Union, Dict, List, Callable, Optional
import torch
import numpy as np

def _transcript_to_tokens(
    transcript: str, 
    token_dictionary: Dict[str, int],
    blank_token: str = "|",
    normalize_fn: Optional[Callable[[str], str]] = None
) -> list[int]:
    """
    Convert transcripts to Token IDS

    Args:
        transcript: Input String
        token_dictionary: Mapping of token to id
        blank_token: What blank token character does the CTC system use?
        normalize_fn: Optional function to make text compatible with token_dictionary
    """

    ### Remove any white spaces ###
    transcript = transcript.replace("\n", "").strip()   

    ### Apply normalize function if available ###
    if normalize_fn is not None:
        transcript = normalize_fn(transcript)

    ### Separate each character with delimiter ###
    return [token_dictionary[c] for c in blank_token.join(transcript.split())]

def _to_numpy(emission) -> np.ndarray:
    if isinstance(emission, torch.Tensor):
        return emission.detach().cpu().float().contiguous().numpy().copy()  # .copy() forces ownership
    arr = np.asarray(emission, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"] or not arr.flags["OWNDATA"]:
        arr = np.array(arr)  # copy, not just contiguous view
    return arr



