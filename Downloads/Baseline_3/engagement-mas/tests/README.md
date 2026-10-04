# Tests README

This directory contains unit and integration tests for the **Engagement‑MAS** project. All tests can be run with the standard `pytest` command.

## Available test files
| Test file | Purpose |
|-----------|---------|
| `test_determinism.py` | Ensures that each perception block (`MacroBlock`, `RPPGBlock`, `MERBlock`) produces identical output when invoked twice on the same clip. |
| `test_gate.py` | Validates the synchronization gate’s behavior: early release when all modalities arrive, timeout handling for missing modalities, discarding late arrivals, and ignoring duplicate results. |
| `test_fusion.py` | Checks that the three fusion pipelines (`EarlyFusion`, `StaticLateFusion`, `AgenticFusion`) are deterministic – repeated runs on the same bundle give the same engagement probabilities. |

## Running the tests
1. Activate the appropriate Conda environment (Mac for development/inference, Cloud for GPU training). Example for Mac:
   ```bash
   conda activate engagement-mas-mac
   ```
2. Install the testing dependency (already added to the environment):
   ```bash
   pip install pytest
   ```
3. From the repository root, execute:
   ```bash
   pytest -q tests
   ```
   The `-q` flag gives a concise output; you should see all tests passing.

## Test data
A small sample video (`data/sample_clip.mp4`) is required for the deterministic block tests. If you do not have one, you can place any short MP4 clip (≈1 s) at that path – the tests only check for repeatability, not visual quality.

## Adding new tests
When adding further tests, follow the same pattern:
- Place the file under `tests/`.
- Use plain Python `assert` statements.
- Keep each test independent (no shared mutable state).
- Run `pytest` to ensure they pass before committing.
