# CAPE

This repository contains the official implementation of **CAPE: Context-Aware Pruning for Ordering-Based Causal Discovery**.

> Junghyo Sohn, Sujeong Song, Wootaek Jeong, Yeajin Shon, and Heung-Il Suk,  
> “Context-Aware Pruning for Ordering-Based Causal Discovery,”  
> *Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM 2026)*,  
> Rome, Italy, November 7-11, 2026.

CAPE is a plug-in pruning method for ordering-based causal discovery. It evaluates each candidate parent together with its current co-parents and removes edges that do not provide enough predictive evidence.

## Key Features

- Context-aware edge scoring
- Adaptive MDL-inspired pruning threshold
- Hierarchical group pruning for faster inference
- Compatible with multiple ordering methods and predictors

Supported ordering methods: `cam`, `score`, `nogam`, `diffan`, `caps`, `scino`

Supported pruning methods: `cape`, `cape-atomic`, `cam`

## Repository Structure

```text
.
├── configs/
│   └── default.yaml          # Default experiment configuration
├── ordering/
│   ├── cam.py                # CAM ordering
│   ├── caps.py               # CaPS ordering
│   ├── diffan.py             # DiffAN ordering
│   ├── nogam.py              # NoGAM ordering
│   ├── scino.py              # SciNO ordering
│   └── score.py              # SCORE ordering
├── pruning/
│   ├── cape.py               # Hierarchical CAPE pruning
│   ├── cape_atomic.py        # Edge-wise CAPE pruning
│   ├── cam_pruning.py        # CAM pruning wrapper
│   ├── pruning_R_files/      # R scripts for CAM pruning
│   └── TabPFN/               # TabPFN dependency (added during setup)
├── dag_simulation.py         # Synthetic DAG and data generation
├── dataset.py                # Dataset loading and preprocessing
├── main.py                   # Main experiment entry point
├── model_runner.py           # Ordering and pruning method registry
├── utils.py                  # Shared utility functions
└── README.md
```

The following directories are not included by default:

- `pruning/TabPFN/`: created when installing TabPFN
- `Datasets/`: used for external datasets
- `results/`: created automatically after running an experiment

## Installation

Python 3.10 or later is required. A CUDA-capable GPU is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy networkx pyyaml scikit-learn tqdm \
    torch gcastle cdt pgmpy pygam
```

### TabPFN

TabPFN is the default predictor used by CAPE.

```bash
git clone --branch v6.4.1 --depth 1 https://github.com/PriorLabs/TabPFN.git pruning/TabPFN
python -m pip install -e pruning/TabPFN
```

Place the TabPFN v2.5 regressor checkpoint at:

```text
pruning/TabPFN/checkpoints/tabpfn-v2.5-regressor-v2.5_default.ckpt
```

### R Dependencies

The evaluation code uses the R package `SID`:

```r
install.packages("BiocManager")
BiocManager::install("SID")
```

The `cam` pruning baseline additionally requires `mgcv`:

```r
install.packages("mgcv")
```

## Running an Experiment

Edit `configs/default.yaml`, then run the following command from the repository root:

```bash
python main.py --config configs/default.yaml
```

The main configuration options are:

```yaml
general:
  device: cuda:0
  dataset: SynER4
  num_nodes: 10
  num_samples: 2000
  runs: 10

ordering:
  model: score

pruning:
  method: cape
  cape:
    predictor: tabpfn
```

## Datasets

Available synthetic datasets:

- `SynER{k}`: Erdős-Rényi graph
- `SynSF{k}`: Scale-Free graph

The code also supports:

- `sachs`
- `magic-niab`
- `magic-irri`
- `physics`

The Physics dataset must be placed under:

```text
Datasets/physics_generation/
```

## Results

Results are saved automatically under `results/`.

Each result directory contains:

- Experiment configuration
- Console output log
- Evaluation metrics in CSV format
