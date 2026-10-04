# Configurations

All YAML configuration files live here:

* **datasets.yaml** – dataset‑specific parameters (root path, CSV names, label schema).
* **training.yaml** – hyper‑parameters for training the Macro block (batch size, learning‑rate, epochs, LoRA settings, loss weighting, etc.).
* **gate.yaml** – synchronization gate settings (timeout, buffer size, logging options).
* **fusion.yaml** – fusion‑strategy parameters, modality‑weight thresholds, reasoning‑engine config.

Each file is documented at the top with a description of every field. The code reads them with `hydra`/`omegaconf` so you can override values from the command line.
