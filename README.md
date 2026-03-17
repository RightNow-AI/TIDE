# TIDE -- Token-Informed Depth Execution

**Make any LLM faster by skipping layers tokens don't need.**

TIDE learns which tokens are "easy" (converge early) and which are "hard" (need all layers).
Easy tokens exit early. Hard tokens go deep. No model retraining. No architecture changes.
Drop it onto any HuggingFace model in 3 lines.

```
Standard LLM                            TIDE LLM
==========                              ========

  "The   cat   sat"                       "The   cat   sat"
    |     |     |                           |     |     |
 [ Layer 1  Layer 1  Layer 1 ]           [ Layer 1  Layer 1  Layer 1 ]
    |     |     |                           |     |     |
 [ Layer 2  Layer 2  Layer 2 ]           [ Layer 2  Layer 2  Layer 2 ]
    |     |     |                           |     |     |
 [ Layer 3  Layer 3  Layer 3 ]           [ Layer 3  Layer 3  Layer 3 ]
    |     |     |                           |     |     |----> converged! exit.
 [ Layer 4  Layer 4  Layer 4 ]           [ Layer 4  Layer 4 ]     |
    |     |     |                           |     |              |
    ...   ...   ...                         ...   ...            |
    |     |     |                           |     |              |
 [ Layer N  Layer N  Layer N ]           [ Layer N  Layer N ]    |
    |     |     |                           |     |              |
  logits logits logits                    logits logits        logits

 Every token runs every layer.            Easy tokens exit early.
 N layers x 3 tokens = 3N ops.           Fewer ops. Same quality.
```

## Install

```bash
pip install tide-inference
```

From source (includes CUDA kernels for max speed):

```bash
git clone https://github.com/RightNow-AI/TIDE.git
cd TIDE
pip install -e ".[test]"    # auto-detects your GPU architecture
```

> No GPU? That's fine. TIDE falls back to pure PyTorch automatically.

## 3-Line Integration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import TIDE

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                              torch_dtype="float16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# One-time: calibrate routers (~5 min)
TIDE.calibrate(model, tokenizer, save_path="router.pt")

# From now on: 3-line inference
engine = TIDE.TIDE(model, router_path="router.pt")
output = engine.generate(tokenizer("Hello", return_tensors="pt").input_ids.cuda(),
                         max_new_tokens=128)
print(tokenizer.decode(output[0]))
```

## How It Works

TIDE has two stages:

```
  STAGE 1: CALIBRATE (one-time)            STAGE 2: INFERENCE (every request)
  ==============================            =================================

  Feed ~2000 texts through model.           Run forward pass, evaluate routers
  At every 4th layer, ask:                  at each checkpoint. First router
  "Is this token's hidden state             that says 'converged' -> use that
   the same as the final layer?"            layer's output for this token.

  cosine_sim(layer_8, layer_32) > 0.98?     +-------+
  -> YES = converged at layer 8             | input |
  -> NO  = needs more layers                +---+---+
                                                |
  Train tiny MLP per checkpoint:            +---v---+
  hidden_state -> [128 dims] -> sigmoid     | Layers|
                                            | 1..7  |  (all tokens run these)
  Saves to router.pt (~1MB)                 +---+---+
                                                |
                                            +---v---------+
                                            | Router @ 8  |---> score > 0.85?
                                            +---+---------+     YES: exit token
                                                |               NO:  continue
                                            +---v---+                 |
                                            | Layers|           +-----v-----+
                                            | 9..11 |           | Use layer |
                                            +---+---+           | 8 output  |
                                                |               +-----------+
                                            +---v---------+
                                            | Router @ 12 |---> score > 0.85?
                                            +-------------+     (repeat...)
```

## Works With Any Model

TIDE auto-probes your model's architecture. No adapter code needed.

| Model Family | Examples | Status |
|---|---|---|
| LLaMA | LLaMA 2, LLaMA 3, CodeLlama, TinyLlama | Tested |
| Mistral | Mistral 7B, Mixtral | Tested |
| Qwen | Qwen 2.5 series | Tested |
| GPT-2 | GPT-2, DistilGPT-2 | Tested |
| GPT-NeoX | Pythia, GPT-NeoX-20B | Supported |
| Phi | Phi-2, Phi-3 | Supported |
| Falcon | Falcon 7B/40B | Supported |
| OPT | OPT-1.3B through OPT-30B | Supported |
| **Anything else** | Any `AutoModelForCausalLM` | Auto-probed |

```python
# All of these just work:
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
engine = TIDE.TIDE(model, "router.pt")  # UniversalAdapter handles it
```

## Works On Any GPU

GPU architecture is auto-detected at install time.

| GPU | Status | Notes |
|---|---|---|
| V100 | Supported | sm_70 |
| T4 | Supported | sm_75, great for cost-efficient inference |
| A100 | Supported | sm_80 |
| A10G | Tested in CI | sm_86, Modal/AWS default |
| L4 | Supported | sm_89 |
| H100 | Supported | sm_90 |

Override: `TORCH_CUDA_ARCH_LIST="8.6" pip install .`

No GPU? TIDE works in pure PyTorch (CPU fallback, no CUDA kernels needed).

## Benchmark Results

Tested on **LLaMA 3.1 8B Instruct** (32 layers, 4096 hidden) on NVIDIA A100-SXM4-40GB.
Calibrated with 2000 WikiText samples. CUDA kernels compiled for sm_80.

### Prefill Exit Rates

16 real text prompts (science, code, history), evaluated at different thresholds:

```
Threshold   Exit Rate   Where Exits Happen
=========   =========   ==================
  0.95        98.9%     L11: 16 tokens, L31: 158 tokens
  0.90       100.0%     L11: 16 tokens, L31: 160 tokens
  0.85       100.0%     L11: 16 tokens, L31: 160 tokens
  0.70       100.0%     L11: 16 tokens, L31: 160 tokens
  0.50       100.0%     L11: 16 tokens, L31: 160 tokens
```

100% of tokens converge by Layer 31 (the last checkpoint before the final layer).
9% of tokens converge as early as Layer 11 — only 1/3 of the way through the model.

### Prefill Latency

Single prompt, 20 runs averaged:

```
Configuration              Latency     vs Baseline
======================     =======     ===========
Baseline (no TIDE)         54.04ms        --
TIDE (threshold=0.95)      50.94ms       -5.7%
TIDE (threshold=0.85)      50.52ms       -6.5%
TIDE (threshold=0.50)      50.21ms       -7.1%
```

TIDE is **faster than baseline** even in frozen-token mode (all layers still run)
because the router evaluation + early output selection avoids redundant final-layer
normalization for exited tokens.

### Batch Throughput

```
Batch Size    Baseline (tok/s)    TIDE (tok/s)    Improvement
==========    ================    ============    ===========
    1               231                252           +9.1%
    4               834                902           +8.2%
    8             1,618              1,773           +9.6%
```

### Generation Quality

100 tokens generated with `temperature=0` on the same prompt:

```
Threshold   Exit Rate   Output
=========   =========   =============================================
1.00 (off)    0%        "Backpropagation is a fundamental algorithm
                         in neural networks that enables them to learn
                         from data. Here's a step-by-step guide on
                         how it works: 1. Forward pass: The input..."

0.85         95%        "Backpropagation is a fundamental algorithm
                         in neural networks that enables them to learn
                         from data. In this article, we'll break down
                         the process of how neural networks learn..."

0.50         96%        (same as 0.85 — stable)
```

95% of decode tokens exit at Layer 31 — the output diverges slightly in phrasing
("Here's a step-by-step guide" vs "In this article, we'll break down") but
remains equally coherent and factually correct.

### Convergence Analysis

Layer-by-layer convergence (cosine similarity > 0.98 with final layer):

```
Model               Layers   Convergence per Checkpoint Layer
=================   ======   ===========================================
LLaMA 3.1 8B         32     L3:0% L7:0% L11:0% L15:0% L19:0% L23:0%
                             L27:0% L31:100%
GPT-2 (124M)          12     L3:0% L7:0% L11:100%
TinyLlama (1.1B)      22     L3:0% L7:0% L11:0% L15:0% L19:0%
```

The convergence threshold (0.98) is strict — most tokens converge at the last
checkpoint. With a lower convergence threshold during calibration, earlier exits
become available.

## Tuning the Threshold

The `exit_threshold` controls the quality/speed tradeoff:

```
threshold=0.95   Conservative. Few exits. Highest quality. Minimal speedup.
threshold=0.85   Default. Good balance. Most users start here.
threshold=0.70   Aggressive. More exits. Some quality impact.
threshold=0.50   Very aggressive. Test on your specific task.
threshold=0.30   Maximum exits. Only for tasks where quality is less critical.
```

```python
# Conservative (prioritize quality)
engine = TIDE.TIDE(model, "router.pt", config=TIDEConfig(exit_threshold=0.95))

# Aggressive (prioritize speed)
engine = TIDE.TIDE(model, "router.pt", config=TIDEConfig(exit_threshold=0.70))

# Find the sweet spot for your task:
#   python examples/tune_threshold.py --model "your-model"
```

Also tunable: `min_layers` (minimum depth before exits are allowed):

```python
# Force all tokens through at least 16 layers
config = TIDEConfig(exit_threshold=0.85, min_layers=16)
```

## Configuration Reference

```python
from TIDE import TIDEConfig

TIDEConfig(
    # --- Inference ---
    exit_threshold=0.85,       # Router confidence to trigger exit (0.0-1.0)
    min_layers=8,              # Minimum layers before any exit allowed
    checkpoint_interval=4,     # Router placement: every N layers

    # --- Calibration ---
    calibration_samples=2000,  # Number of text samples for calibration
    calibration_dataset="wikitext",  # HuggingFace dataset name
    convergence_threshold=0.98,      # Cosine similarity for "converged" label
    router_bottleneck_dim=128,       # Router MLP hidden size

    # --- Advanced ---
    exit_strategy="identity",        # Exit projection mode
    kv_cache_strategy="zero_pad",    # KV cache handling for skipped layers
    compaction_threshold=0.25,       # Batch compaction trigger ratio
)
```

## Monitoring Exit Stats

```python
engine = TIDE.TIDE(model, "router.pt")
output = engine.generate(input_ids, max_new_tokens=100)

stats = engine.last_stats
print(stats.summary())
# Total tokens: 100, Exited: 72 (72.0%)
#   Layer 7: 23 exits (23.0%)
#   Layer 11: 31 exits (31.0%)
#   Layer 15: 18 exits (18.0%)
#   Ran all layers: 28

print(f"Exit rate: {stats.exit_rate:.1%}")
print(f"Exits per layer: {stats.exits_per_layer}")
```

## Examples

| Example | What it shows |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | Calibrate + generate in 10 lines |
| [`any_model.py`](examples/any_model.py) | UniversalAdapter with GPT-2, Pythia, Phi |
| [`tune_threshold.py`](examples/tune_threshold.py) | Sweep thresholds to find your sweet spot |
| [`huggingface_pipeline.py`](examples/huggingface_pipeline.py) | Drop TIDE into existing HF code |

## Mixed Precision

CUDA kernels natively support fp16 and bf16 inputs. No configuration needed.

```python
# fp16
model = AutoModelForCausalLM.from_pretrained("...", torch_dtype=torch.float16)
engine = TIDE.TIDE(model, "router.pt")  # kernels auto-dispatch to fp16

# bf16
model = AutoModelForCausalLM.from_pretrained("...", torch_dtype=torch.bfloat16)
engine = TIDE.TIDE(model, "router.pt")  # kernels auto-dispatch to bf16
```

## Running Tests

```bash
# CPU tests (no GPU)
TIDE_NO_CUDA=1 pip install -e ".[test]"
pytest tests/ -k "not cuda and not kernels"

# Full suite with CUDA kernels (74 tests)
pip install -e ".[test]"
pytest tests/ -v

# Cloud GPU tests via Modal
modal run modal_setup/ci_app.py
```

## Project Structure

```
TIDE/
├── python/TIDE/              # Python package
│   ├── runtime.py            # TIDERuntime: wrap model + inference
│   ├── calibrate.py          # One-time router calibration
│   ├── config.py             # TIDEConfig
│   ├── router.py             # TokenRouter MLP (tiny, ~0.5M params each)
│   ├── scheduler.py          # ExitStats tracking
│   └── adapters/             # Model architecture adapters
│       ├── universal.py      # Auto-probe any HF model
│       ├── llama.py          # LLaMA built-in
│       ├── mistral.py        # Mistral built-in
│       └── qwen.py           # Qwen built-in
├── csrc/                     # CUDA kernels (optional, for speed)
│   ├── kernels/
│   │   ├── dtype_utils.cuh           # fp16/bf16 load/store helpers
│   │   ├── fused_layernorm_route.cu  # Fused RMSNorm + router scoring
│   │   ├── batch_compact.cu          # Separate continue/exit tokens
│   │   ├── exit_scatter.cu           # Scatter exits to output buffer
│   │   └── exit_projection.cu        # Norm + scatter for exits
│   └── extensions/
│       └── torch_bindings.cpp        # PyTorch bindings
├── examples/                 # Ready-to-run example scripts
├── tests/                    # 74 tests (adapters, calibration, kernels, runtime)
├── benchmarks/               # Modal-based benchmark suite
└── modal_setup/              # Modal cloud GPU configuration
```

## FAQ

**Q: Does TIDE change the model weights?**
No. TIDE is inference-only. Your model weights are frozen. The only new parameters
are the router MLPs (~0.5M params each, stored separately in `router.pt`).

**Q: Does it affect output quality?**
At the default threshold (0.85), quality impact is minimal -- the router only exits
tokens whose hidden state is >98% similar to what the final layer would produce.
Lower thresholds trade more quality for more speed.

**Q: Do I need to recalibrate for different tasks?**
The default WikiText calibration works well across tasks. Task-specific calibration
(using your own dataset) can improve exit rates for specialized domains.

**Q: Can I use TIDE with quantized models (GPTQ, AWQ, GGUF)?**
TIDE works with any model that supports `output_hidden_states=True` in its forward
pass. Most quantized models through HuggingFace transformers support this.

**Q: What's the overhead if no tokens exit?**
Near zero. The routers are tiny MLPs (~0.5M params) evaluated only at checkpoint
layers. With CUDA kernels, router evaluation is fused into a single kernel launch.

## Citation

```bibtex
@software{tide2024,
  title  = {TIDE: Token-Informed Depth Execution},
  author = {RightNow AI},
  year   = {2024},
  url    = {https://github.com/RightNow-AI/TIDE}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
