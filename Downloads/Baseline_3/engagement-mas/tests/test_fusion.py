# Fusion tests

"""test_fusion.py
Tests that each fusion method is deterministic and produces consistent outputs given the same bundle.
"""

import numpy as np
from pathlib import Path

from fusion.early_fusion import EarlyFusion
from fusion.static_late_fusion import StaticLateFusion
from fusion.agentic_fusion import AgenticFusion

# Build a synthetic bundle that mimics the output of SyncGate
def make_bundle():
    # Dummy predictions are probability vectors (4 classes) per modality
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    bundle = {
        "predictions": {
            "macro": probs,
            "rppg": probs,
            "mer": probs,
        },
        "features": {
            "macro": np.random.rand(768),
            "rppg": np.random.rand(128),
            "mer": np.random.rand(256),
        },
        "quality_metrics": {
            "macro": {"blur": 0.5},
            "rppg": {"snr": 12.0},
            "mer": {"flow_noise": 0.1},
        },
        "availability": {"macro": True, "rppg": True, "mer": True},
    }
    return bundle


def _run_fusion_twice(fusion_cls, config_path="configs/fusion.yaml"):
    fusion = fusion_cls(config_path)
    bundle = make_bundle()
    out1 = fusion.run(bundle)
    out2 = fusion.run(bundle)
    # Compare the engagement probability vectors (key "engagement")
    eps = np.max(np.abs(out1["engagement"] - out2["engagement"]))
    assert eps < 1e-9, f"Fusion {fusion_cls.__name__} not deterministic: max diff {eps}"

def test_early_fusion_deterministic():
    _run_fusion_twice(EarlyFusion)

def test_static_late_fusion_deterministic():
    _run_fusion_twice(StaticLateFusion)

def test_agentic_fusion_deterministic():
    _run_fusion_twice(AgenticFusion)

if __name__ == "__main__":
    test_early_fusion_deterministic()
    test_static_late_fusion_deterministic()
    test_agentic_fusion_deterministic()
    print("All fusion determinism tests passed")
