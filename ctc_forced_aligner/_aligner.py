import warnings
from typing import Callable, Optional
from .py_aligner import align_single_py
from .utils import _to_numpy, _transcript_to_tokens

try:
    from ._ctc_align_cpp import (
        align_single as _cpp_align_single, 
        align_batch as _cpp_align_batch
    )
    _BACKEND = "cpp"
except ImportError:
    warnings.warn(
        "ctc_forced_aligner: C++ extension (_ctc_align_cpp) not found. "
        "Falling back to the pure-Python backend. "
        "Run `pip install -e .` (or `python setup.py build_ext --inplace`) "
        "to build the fast extension.",
        RuntimeWarning,
        stacklevel=2,
    )
    _cpp_align_single = None
    _cpp_align_batch  = None
    _BACKEND = "python"

def backend():
    """return cpp or python"""
    return _BACKEND

def align(
    emission, 
    transcript: str, 
    token_dictionary: dict, 
    blank_token: int,
    normalize_fn: Optional[Callable[[str], str]] = None,
    fast: bool=True, 
    return_trellis: bool = False
):

    """
    Align a single emission matrix to a transcript
    
    Args:
        emission: (num_frames x vocab size) tensor/array of log probabilities
        transcript: plain-text transcript string.
        token_dictionary: character to token_id mappings
        blank_token_id: CTC blank token id
        fast: Use the C++ backend (when available), default True
        return_trellis: Return the (num_frames, num_tokens) trellis alongside spans, 
                        but forced python backend bceause C++ doesnt return this
    """   

    use_cpp = fast and not return_trellis and _cpp_align_single is not None

    if use_cpp:
        emission_np = _to_numpy(emission)
        tokens = _transcript_to_tokens(transcript, token_dictionary, 
                                       blank_token, normalize_fn)
        return _cpp_align_single(emission_np, tokens, token_dictionary[blank_token])
    
    return align_single_py(
        emission, transcript, token_dictionary,
        blank_token, return_trellis=return_trellis
    )

def align_batch(
    emissions: list, 
    transcripts: list[str], 
    token_dictionary: dict, 
    blank_token: int,
    normalize_fn: Optional[Callable[[str], str]] = None,
    fast: bool=True, 
    return_trellis: bool = False
):

    assert len(emissions) == len(transcripts), \
        "each emission should have a cooresponding transcript"

    use_cpp = fast and not return_trellis and _cpp_align_single is not None

    if use_cpp:
        emissions_np = [_to_numpy(e) for e in emissions]
        token_seqs   = [_transcript_to_tokens(t, token_dictionary, blank_token, normalize_fn) for t in transcripts]
        result = _cpp_align_batch(emissions_np, token_seqs, token_dictionary[blank_token])
        del emissions_np
        return result
    
    return [
        align_single_py(
            e, t, token_dictionary, blank_token,
            return_trellis=return_trellis,
        )
        for e, t in zip(emissions, transcripts)
    ]