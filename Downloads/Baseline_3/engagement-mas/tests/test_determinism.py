# Determinism tests

"""test_determinism.py
Tests that each perception block produces identical outputs when run twice on the same clip.
"""

import pathlib
import numpy as np

from blocks.macro.macro_block import MacroBlock
from blocks.rppg.rppg_block import RPPGBlock
from blocks.mer.mer_block import MERBlock

# Choose a small clip for quick testing – assume a sample clip exists at data/sample_clip.mp4
SAMPLE_CLIP = pathlib.Path("data") / "sample_clip.mp4"


def _run_twice(block, clip_path):
    out1 = block.run(str(clip_path))
    out2 = block.run(str(clip_path))
    return out1, out2


def _compare_dicts(d1, d2):
    # Compare numeric entries (features, predictions, quality_metrics) using np.allclose
    for key in ["features", "predictions", "quality_metrics"]:
        if key not in d1 or key not in d2:
            continue
        arr1 = np.asarray(d1[key])
        arr2 = np.asarray(d2[key])
        assert np.allclose(arr1, arr2, atol=1e-8), f"{key} differs"

def test_macro_determinism():
    block = MacroBlock()
    out1, out2 = _run_twice(block, SAMPLE_CLIP)
    _compare_dicts(out1, out2)

def test_rppg_determinism():
    block = RPPGBlock()
    out1, out2 = _run_twice(block, SAMPLE_CLIP)
    _compare_dicts(out1, out2)

def test_mer_determinism():
    block = MERBlock()
    out1, out2 = _run_twice(block, SAMPLE_CLIP)
    _compare_dicts(out1, out2)
