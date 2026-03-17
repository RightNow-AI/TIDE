"""Modal volume definitions for persistent storage."""

import modal

model_cache = modal.Volume.from_name("TIDE-model-cache", create_if_missing=True)
calibration_data = modal.Volume.from_name("TIDE-calibration", create_if_missing=True)
router_weights = modal.Volume.from_name("TIDE-routers", create_if_missing=True)
benchmark_results = modal.Volume.from_name("TIDE-benchmarks", create_if_missing=True)

VOLUME_MOUNTS = {
    "/root/models": model_cache,
    "/root/calibration": calibration_data,
    "/root/routers": router_weights,
    "/root/results": benchmark_results,
}
