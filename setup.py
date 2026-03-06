from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np

ext = Pybind11Extension(
    "ctc_forced_aligner._ctc_align_cpp",
    sources=["ctc_forced_aligner/ctc_align.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-O3",
        "-march=native",   
        "-funroll-loops",
        "-ffast-math",    
    ],
    language="c++",
    cxx_std=17,
)

setup(
    name="ctc_forced_aligner",
    version="0.1.0",
    description="Fast CTC forced alignment with a C++ backend",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
    install_requires=[
        "pybind11>=2.11",
        "numpy",
        "torch",
        "transformers",
        "torchaudio",
        "librosa",
        "pandas",
        "tqdm",
        "accelerate",
        "torchcodec"
    ],
    entry_points={
    "console_scripts": [
        "bulk-align=ctc_forced_aligner.cli:main",
        ],
    },
)