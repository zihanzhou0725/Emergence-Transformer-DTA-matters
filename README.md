# DTA Platform V1

A PyTorch implementation of the V1 **Synchronization Transformer** for numerical experiments on coupled oscillator synchronization and desynchronization.

This repository focuses on the V1 model only: a single-layer Dynamical Temporal Attention (DTA) mechanism that augments traditional network coupling with an attention-based temporal memory term.

## Features

- Single-layer Synchronization Transformer for networked phase oscillators.
- Synchronization and desynchronization control objectives.
- Neighbor-DTA and Self-DTA attention network modes.
- Watts-Strogatz small-world and fully connected network support.
- Training, evaluation, visualization, and baseline comparison scripts.
- Traditional coupling baseline with `alpha approximately 0`.
- Natural memory baseline based on exponential decay memory.
- MATLAB `.mat` export for comparison curves.

## Model Summary

The V1 model simulates a network of `N` phase oscillators. Each oscillator has phase `theta_i`, natural frequency `omega_i`, and is coupled through a spatial network `A`.

The phase update is:

```text
theta_{t+1} = theta_t + (omega + lambda * Im[I_t * exp(-i theta_t)]) * dt + noise
```

The total coupling information is a mixture of traditional spatial coupling and DTA temporal attention:

```text
I_t = (1 - alpha) * spatial_coupling + alpha * attention_coupling
```

where:

- `spatial_coupling = A @ exp(i theta_t) / degree`
- `attention_coupling = A_hat @ M_t / degree_hat`
- `M_t` is computed from the full accumulated phase history using Transformer-style attention.
- `alpha = sigmoid(alpha_param)` controls the mixture between traditional coupling and DTA.

Two attention-network modes are supported:

- `neighbor`: `A_hat = A`, attention information follows the physical network.
- `self`: `A_hat = I`, each oscillator uses its own temporal memory.

## Installation

Recommended environment:

- Python 3.9+
- PyTorch 1.10+

Clone the repository and install dependencies:

```bash
git clone https://github.com/zihanzhou0725/Emergence-Transformer-DTA-matters.git
cd Emergence-Transformer-DTA-matters
pip install -r requirements.txt
```

If you use Conda:

```bash
conda create -n dta-platform python=3.9
conda activate dta-platform
git clone https://github.com/zihanzhou0725/Emergence-Transformer-DTA-matters.git
cd Emergence-Transformer-DTA-matters
pip install -r requirements.txt
```

## Quick Start

Run the demo:

```bash
python demo.py
```

Train a synchronization controller:

```bash
python train.py --config sync
```

Train a desynchronization controller:

```bash
python train.py --config desync
```

Run a DTA vs traditional coupling comparison:

```bash
python compare_with_baseline.py
```

Evaluate a trained checkpoint:

```bash
python evaluate.py --checkpoint results/sync_neighbor_ws/final_model.pt
```

## Project Structure

```text
dta_platform/
├── configs/
│   └── default_config.py          # experiment configurations
├── models/
│   ├── sync_transformer.py        # Synchronization Transformer
│   └── controller.py              # sync/desync loss wrappers
├── utils/
│   ├── networks.py                # graph generation utilities
│   ├── metrics.py                 # order parameter and metrics
│   ├── trainer.py                 # training loop
│   └── visualization.py           # plotting utilities
├── train.py                       # train models
├── evaluate.py                    # evaluate trained checkpoints
├── demo.py                        # runnable demo
├── compare_with_baseline.py       # DTA vs traditional coupling
├── memory_compare_with_baseline.py # natural memory vs traditional coupling
├── generate_trained_trajectory.py # generate R-t curves from a checkpoint
├── visualize_attention_weights.py # inspect learned W_Q/W_K
├── requirements.txt
└── README.md
```

## Training

The main training entry point is:

```bash
python train.py --config <config_name>
```

Available configurations:

| Config | Task | Attention Network | Default Save Directory |
| --- | --- | --- | --- |
| `sync` | Synchronization, `R -> 1` | `neighbor`, `A_hat = A` | `results/sync_neighbor_ws` |
| `sync_self` | Synchronization, `R -> 1` | `self`, `A_hat = I` | `results/sync_self_ws` |
| `desync` | Desynchronization, `R -> 0` | `self`, `A_hat = I` | `results/desync_self_ws` |
| `desync_neighbor` | Desynchronization, `R -> 0` | `neighbor`, `A_hat = A` | `results/desync_neighbor_ws` |
| `test` | Small quick test | configuration dependent | `results/test_ws` |

Useful command-line options:

```bash
python train.py \
  --config sync \
  --n_oscillators 100 \
  --epochs 100 \
  --lr 0.001 \
  --seed 42 \
  --save_dir results/my_sync_run
```

Training outputs:

- `final_model.pt`: final checkpoint
- `checkpoint_epoch_*.pt`: periodic checkpoints
- `history.json`: loss and order-parameter history
- `training_curves.png`: training visualization

## Configuration

Core parameters are defined in `configs/default_config.py`.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `NETWORK_TYPE` | `ws` or `fc` | `ws` |
| `N_OSCILLATORS` | number of oscillators | `100` |
| `K_NEIGHBORS` | WS nearest neighbors | `4` |
| `REWIRING_PROB` | WS rewiring probability | `0.1` |
| `NATURAL_FREQ_STD` | std of natural frequencies | `0.1` |
| `D_MODEL` | attention feature dimension | `32` |
| `COUPLING_STRENGTH` | coupling strength `lambda` | `1.0` |
| `NOISE_STRENGTH` | phase noise strength | `0.05` |
| `ALPHA_INIT` | raw alpha parameter before sigmoid | `0.0` |
| `N_EPOCHS` | training epochs | `100` |
| `N_STEPS` | simulation steps per episode | `200` |
| `LEARNING_RATE` | optimizer learning rate | `1e-3` |

Note: the actual mixture coefficient is `sigmoid(ALPHA_INIT)`. For example, `ALPHA_INIT = 0.0` gives `alpha = 0.5`.

## Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
  --checkpoint results/sync_neighbor_ws/final_model.pt \
  --n_trials 20 \
  --n_steps 200 \
  --save_dir results/evaluation_sync
```

The evaluator reports:

- success rate
- final order parameter
- convergence time
- stability estimate

It also saves plots for the first evaluation trial.

## Baseline Comparison

Compare a trained DTA model against traditional coupling by setting `alpha approximately 0` in a copied model:

```bash
python compare_with_baseline.py
```

The script uses the same network, natural frequencies, and model parameters from the checkpoint, then compares:

- trained DTA model
- traditional coupling baseline

Outputs include:

- per-case comparison plots
- average comparison plot
- MATLAB `.mat` data file

## Natural Memory Comparison

The natural memory baseline uses an exponential memory state:

```text
M_{t+1} = M_t + beta * dt * (exp(i theta_t) - M_t)
```

Run:

```bash
python memory_compare_with_baseline.py \
  --checkpoint results/sync_self_ws/final_model.pt \
  --memory_alpha 0.5 \
  --memory_beta 0.01 \
  --n_steps 100000 \
  --n_cases 100
```

This compares:

- traditional coupling, `memory_alpha = 0`
- natural memory coupling, `memory_alpha > 0`

Use smaller values for a quick smoke test:

```bash
python memory_compare_with_baseline.py --n_steps 500 --n_cases 2
```

## Visualizing Learned Attention Weights

Inspect learned `W_Q` and `W_K` matrices from a checkpoint:

```bash
python visualize_attention_weights.py --checkpoint results/sync_neighbor_ws/final_model.pt
```

This is useful for understanding how the V1 DTA model encodes temporal attention.

## Reproducibility

Training scripts set PyTorch and NumPy random seeds through `--seed`.

Checkpoints save:

- model weights
- optimizer state
- task type
- attention type
- network type
- spatial network
- attention network
- natural frequencies
- coupling strength

For reproducible comparisons, use the checkpoint-based comparison scripts rather than regenerating network parameters manually.

## Common Workflows

Train and compare a neighbor-DTA synchronization model:

```bash
python train.py --config sync --epochs 100
python compare_with_baseline.py
```

Train and compare a self-DTA synchronization model:

```bash
python train.py --config sync_self --epochs 100
python compare_with_baseline.py
```

Run a quick development test:

```bash
python train.py --config test
python evaluate.py --checkpoint results/test_ws/final_model.pt --n_trials 3 --n_steps 100
```

## Troubleshooting

If PyTorch or OpenMP prints duplicate library warnings on Windows, the provided scripts already set:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

If CUDA memory is limited, run comparison scripts on CPU or reduce:

- `N_OSCILLATORS`
- `N_STEPS`
- `N_CASES`

If a checkpoint path does not exist, train the corresponding model first or pass the correct path with `--checkpoint`.

## Citation

This code is inspired by the Synchronization Transformer / Dynamical Temporal Attention formulation for coupled oscillator dynamics.

If you use this repository in academic work, please cite the original paper “Emergence Transformer: Dynamical Temporal Attention Matters”.

## License

MIT License
