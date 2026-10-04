import os
import time
import yaml
import logging
import threading
from typing import Dict, Any, Optional

class SyncGate:
    """Synchronization gate for aggregating modality results.

    - First result for a ``clip_id`` opens a collection window and starts ``timeout`` seconds.
    - If all three modalities (macro, rppg, mer) arrive early, the bundle is emitted immediately.
    - If the timeout expires, the bundle is emitted with whatever modalities have arrived.
    - Late arrivals after a bundle has been emitted are logged and discarded.
    - Duplicate results for the same modality are ignored (first kept).
    - Results with ``status == 'failed'`` are treated as absent.
    """

    MODALITIES = {"macro", "rppg", "mer"}

    def __init__(self, config_path: str = "configs/gate.yaml", timeout: float = None):
        # Load configuration (timeout in seconds)
        cfg_path = os.path.abspath(config_path)
        if os.path.isfile(cfg_path):
            cfg = yaml.safe_load(open(cfg_path))
            self.timeout = timeout if timeout is not None else cfg.get("timeout", 60)
            log_path = cfg.get("log_path", "gate.log")
        else:
            self.timeout = timeout if timeout is not None else 60
            log_path = "gate.log"
        log_dir = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s", force=True)
        self.logger = logging.getLogger("SyncGate")
        # Internal state: clip_id -> {"start_time": float, "results": {modality: result}}
        self._pending: Dict[str, Dict[str, Any]] = {}
        # Store emitted bundles for reference (useful for tests)
        self._emitted_bundles: Dict[str, Dict[str, Any]] = {}
        # Track retrieved bundles for check_and_release
        self._retrieved_bundles: set = set()
        # Late results log
        self.late_results: list[Dict[str, Any]] = []
        self._finalized: bool = False
        # Thread‑safety lock
        self._lock = threading.Lock()

    def _now(self) -> float:
        """Current time in seconds. Separated for testing/mocking."""
        return time.time()

    def _create_bundle(self, clip_id: str) -> Dict[str, Any]:
        entry = self._pending.pop(clip_id)
        results = entry["results"]
        # Build availability mask
        availability = {mod: (mod in results and results[mod].get("status") == "success")
                        for mod in self.MODALITIES}
        # Assemble bundle
        bundle = {
            "clip_id": clip_id,
            "features": {mod: results[mod]["features"] for mod in results if "features" in results[mod]},
            "predictions": {mod: results[mod]["predictions"] for mod in results if "predictions" in results[mod]},
            "quality_metrics": {mod: results[mod]["quality_metrics"] for mod in results if "quality_metrics" in results[mod]},
            "availability": availability,
            "timestamps": {mod: results[mod]["timestamps"] for mod in results if "timestamps" in results[mod]},
        }
        self._emitted_bundles[clip_id] = bundle
        self.logger.info(f"Bundle emitted for clip {clip_id}: modalities {list(results.keys())}")
        return bundle

    def receive(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Receive a modality result.

        Parameters
        ----------
        result: dict
            Must contain at least ``clip_id`` and ``modality`` keys.

        Returns
        -------
        dict or None
            Returns a bundle dict when the collection window closes (early or timeout),
            otherwise returns ``None``.
        """
        clip_id = result.get("clip_id")
        modality = result.get("modality")
        if not clip_id or not modality:
            raise ValueError("Result must contain 'clip_id' and 'modality'.")
        if modality not in self.MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}. Expected one of {self.MODALITIES}")

        with self._lock:
            # If we already emitted a bundle for this clip, treat as late result
            if clip_id in self._emitted_bundles:
                self.late_results.append(result)
                self.logger.warning(f"Late result received for clip {clip_id}, modality {modality}. Discarded.")
                return None

            # Initialize pending entry if first arrival
            if clip_id not in self._pending:
                self._pending[clip_id] = {"start_time": self._now(), "results": {}}
                self.logger.info(f"First result for clip {clip_id} (modality {modality}) – window opened.")

            entry = self._pending[clip_id]
            # Duplicate modality check
            if modality in entry["results"]:
                self.logger.warning(f"Duplicate result for clip {clip_id}, modality {modality}. Keeping first.")
                return None

            # Treat failed status as absent – we still store it to log later but do not count as present
            if result.get("status") != "success":
                self.logger.info(f"Result for clip {clip_id}, modality {modality} failed with status: {result.get('status')}. Ignored.")
                entry["results"][modality] = result
            else:
                entry["results"][modality] = result

            # Early release if we now have all three successful modalities
            successful = [mod for mod, r in entry["results"].items() if r.get("status") == "success"]
            if self.MODALITIES.issubset(set(successful)):
                return self._create_bundle(clip_id)
            return None

    def check_and_release(self, clip_id: str, force: bool = False) -> Optional[Dict[str, Any]]:
        """Check if clip bundle can be released or was already emitted."""
        with self._lock:
            if clip_id in self._pending:
                entry = self._pending[clip_id]
                now = self._now()
                successful = [mod for mod, r in entry["results"].items() if r.get("status") == "success"]
                if force or self._finalized or self.MODALITIES.issubset(set(successful)) or (now - entry["start_time"] >= self.timeout):
                    bundle = self._create_bundle(clip_id)
                    self._retrieved_bundles.add(clip_id)
                    return bundle
                return None

            if clip_id in self._emitted_bundles:
                if clip_id not in self._retrieved_bundles:
                    self._retrieved_bundles.add(clip_id)
                    return self._emitted_bundles[clip_id]
                return None

            return None

    def finalize(self):
        """Mark gate as finalized so all pending clips are eligible for release."""
        with self._lock:
            self._finalized = True

    def pending_ids(self) -> list[str]:
        """List of pending clip IDs."""
        with self._lock:
            return list(self._pending.keys())

    def check_timeout(self) -> list[Dict[str, Any]]:
        """Force release of any pending bundles whose timeout has expired.

        Returns a list of bundles emitted during this call.
        """
        now = self._now()
        emitted = []
        expired_clip_ids = []
        with self._lock:
            for cid, entry in self._pending.items():
                if now - entry["start_time"] >= self.timeout:
                    expired_clip_ids.append(cid)
            for cid in expired_clip_ids:
                bundle = self._create_bundle(cid)
                self._retrieved_bundles.add(cid)
                emitted.append(bundle)
        return emitted

    # Helper for unit tests to inspect internal state (not part of public API)
    def _reset_state(self):
        with self._lock:
            self._pending.clear()
            self._emitted_bundles.clear()
            self._retrieved_bundles.clear()
            self.late_results.clear()
            self._finalized = False
