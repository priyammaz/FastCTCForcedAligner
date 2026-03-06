from __future__ import annotations

import warnings
from typing import Union, Dict, List, Callable, Optional
import torch
import numpy as np
from ._aligner import align, align_batch, backend
from .py_aligner import align_single_py
from .bulk_aligner import BulkAligner, CTCEmissionExtractor, load_for_inference, get_word_alignments

__all__ = [
    "align", 
    "align_batch", 
    "align_single_py", 
    "backend", 
    "BulkAligner", 
    "CTCEmissionExtractor",
    "load_for_inference",
    "get_word_alignments"
]
