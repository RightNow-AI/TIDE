# Contributing to TIDE

Thanks for your interest in TIDE! Here's how to contribute.

## Development Setup

```bash
git clone https://github.com/RightNowAI/TIDE.git
cd TIDE
pip install -e ".[test]"    # installs with CUDA kernels if GPU available
```

No GPU locally? Use `TIDE_NO_CUDA=1 pip install -e ".[test]"` for CPU-only.

## Running Tests

```bash
# CPU-only tests (fast, no GPU)
pytest tests/ -k "not cuda and not kernels" -v

# Full suite (requires CUDA GPU)
pytest tests/ -v

# Cloud GPU tests
modal run modal_setup/ci_app.py
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting:

```bash
pip install ruff
ruff check python/ tests/
```

## Project Layout

- `python/TIDE/` -- Python package (runtime, calibration, adapters)
- `csrc/` -- CUDA kernels (C++/CUDA)
- `tests/` -- Test suite
- `examples/` -- Example scripts
- `modal_setup/` -- Cloud GPU infrastructure
- `benchmarks/` -- Performance benchmarks

## Adding Support for a New Model Architecture

Most models work automatically via `UniversalAdapter`. If your model doesn't,
add a built-in adapter:

1. Create `python/TIDE/adapters/mymodel.py` implementing `BaseAdapter`
2. Register it in `python/TIDE/adapters/auto.py` under `ADAPTER_REGISTRY`
3. Add a test in `tests/test_adapters.py`

## CUDA Kernel Development

Kernels are in `csrc/kernels/`. All kernels:

- Are templated on `scalar_t` (float, __half, __nv_bfloat16)
- Use `dtype_utils.cuh` for load/store helpers
- Expose named C entry points (not templates) for linking
- Accumulate in float32 for numerical stability

After modifying kernels, rebuild: `pip install -e .`

## Pull Requests

1. Fork and create a branch
2. Make your changes
3. Run `pytest tests/ -v` (at least CPU tests)
4. Run `ruff check python/ tests/`
5. Open a PR with a clear description

## Reporting Issues

Please include:
- Model name and size
- GPU type
- PyTorch and transformers versions
- Full error traceback
