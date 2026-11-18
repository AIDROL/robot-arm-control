"""Atomic observed-state archives; checkpoints never imply acceptance."""
from pathlib import Path

import numpy as np


def atomic_npz(path, **arrays):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,**arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
