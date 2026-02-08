# TFairCRS: Towards Fairness in Conversational Recommender Systems

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Overview

**TFairCRS** is a research project that addresses fairness issues in Conversational Recommender Systems (CRS). This work is built on top of the [CRSLab](https://github.com/RUCAIBox/CRSLab) framework and introduces novel approaches to ensure fair and unbiased recommendations in conversational settings.

### Key Features

- **Fairness-aware recommendation**: Incorporates fairness constraints into the conversational recommendation process
- **Comprehensive evaluation**: Evaluates both recommendation quality and fairness metrics
- **Multiple datasets**: Supports evaluation on standard CRS benchmarks including ReDial, TG-ReDial, and INSPIRED
- **Extensible framework**: Built on CRSLab for easy integration and experimentation

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Method](#method)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

## Installation

### Requirements

- Python 3.6 or later
- PyTorch 1.4.0 or later
- CUDA 9.2 or later (for GPU support)

### Step 1: Install PyTorch

Install PyTorch according to your CUDA version. For example:

```bash
# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Verify GPU availability (if using GPU):

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Step 2: Install PyTorch Geometric

Check your PyTorch and CUDA versions:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

Install PyTorch Geometric components:

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

Replace `${TORCH}` and `${CUDA}` with your versions (e.g., `1.6.0` and `cu101`).

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run TFairCRS with default configuration:

```bash
# CPU
python run_crslab.py --config config/crs/kgsf/redial.yaml

# Single GPU
python run_crslab.py --config config/crs/kgsf/redial.yaml --gpu 0

# Multiple GPUs
python run_crslab.py --config config/crs/kgsf/redial.yaml --gpu 0,1
```

### Available Datasets

You can run experiments on different datasets:

```bash
# ReDial dataset
python run_crslab.py --config config/crs/kgsf/redial.yaml --gpu 0

# TG-ReDial dataset
python run_crslab.py --config config/crs/kgsf/tgredial.yaml --gpu 0

# INSPIRED dataset
python run_crslab.py --config config/crs/kgsf/inspired.yaml --gpu 0
```

### Save and Restore Models

```bash
# Save preprocessed data and trained model
python run_crslab.py --config config/crs/kgsf/redial.yaml --save_data --save_system --gpu 0

# Restore from saved files
python run_crslab.py --config config/crs/kgsf/redial.yaml --restore_data --restore_system --gpu 0
```

### Debug Mode

Use validation set for quick debugging:

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml --debug --gpu 0
```

### Interactive Mode

Interact with the trained model:

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml --interact --restore_system --gpu 0
```

### Monitor Training with TensorBoard

```bash
# Enable tensorboard logging
python run_crslab.py --config config/crs/kgsf/redial.yaml --tensorboard --gpu 0

# In another terminal
tensorboard --logdir=./runs
```

Then visit http://localhost:6006 to monitor training.

### Command Line Arguments

- `--config` / `-c`: Path to configuration file (required)
- `--gpu` / `-g`: GPU id(s) to use (default: "-1" for CPU)
- `--save_data` / `-sd`: Save preprocessed dataset
- `--restore_data` / `-rd`: Restore preprocessed dataset
- `--save_system` / `-ss`: Save trained model
- `--restore_system` / `-rs`: Restore trained model
- `--debug` / `-d`: Debug mode using validation set
- `--interact` / `-i`: Interactive mode
- `--tensorboard` / `-tb`: Enable tensorboard logging

## Method

### Problem Statement

Traditional conversational recommender systems often suffer from fairness issues, where certain groups of items or users may receive disproportionate treatment. TFairCRS addresses these challenges by incorporating fairness constraints into the recommendation process.

### Approach

Our method introduces fairness-aware mechanisms that:

1. **Fairness-aware User Modeling**: Ensures balanced representation across different user groups
2. **Fair Item Exposure**: Guarantees equitable visibility for items across different categories
3. **Bias Mitigation**: Reduces demographic and popularity biases in recommendations
4. **Multi-objective Optimization**: Balances recommendation accuracy with fairness metrics

### Architecture

TFairCRS builds upon existing CRS models (KGSF, KBRD, TG-ReDial, etc.) and extends them with fairness-aware components.

## Datasets

We evaluate TFairCRS on multiple standard CRS datasets:

| Dataset | Dialogs | Utterances | Domain | Knowledge Graph |
|---------|---------|------------|--------|-----------------|
| [ReDial](https://redialdata.github.io/website/) | 10,006 | 182,150 | Movie | DBpedia + ConceptNet |
| [TG-ReDial](https://github.com/RUCAIBox/TG-ReDial) | 10,000 | 129,392 | Movie | CN-DBpedia + HowNet |
| [INSPIRED](https://github.com/sweetpeach/Inspired) | 1,001 | 35,811 | Movie | DBpedia + ConceptNet |

## Experiments

### Evaluation Metrics

We evaluate TFairCRS on both **recommendation quality** and **fairness** metrics:

**Recommendation Metrics:**
- Hit@{1, 10, 50}
- MRR@{1, 10, 50}
- NDCG@{1, 10, 50}

**Conversation Metrics:**
- BLEU-{1, 2, 3, 4}
- Distinct-{1, 2, 3, 4}
- Embedding metrics (Average, Extreme, Greedy)
- Perplexity (PPL)

**Fairness Metrics:**
- Demographic parity
- Equal opportunity
- Exposure fairness
- Statistical parity difference

### Results

*Results will be updated upon publication.*

## Project Structure

```
TFairCRS/
├── config/              # Configuration files
│   ├── crs/            # CRS model configs
│   ├── conversation/   # Conversation model configs
│   └── recommendation/ # Recommendation model configs
├── data/               # Dataset directory
├── run_crslab.py       # Main entry point
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Citation

## Acknowledgments

This project is built on top of [CRSLab](https://github.com/RUCAIBox/CRSLab). We thank the CRSLab team for their excellent framework.

```bibtex
@article{crslab,
    title={CRSLab: An Open-Source Toolkit for Building Conversational Recommender System},
    author={Kun Zhou, Xiaolei Wang, Yuanhang Zhou, Chenzhan Shang, Yuan Cheng, Wayne Xin Zhao, Yaliang Li, Ji-Rong Wen},
    year={2021},
    journal={arXiv preprint arXiv:2101.00939}
}
```

## License

This project is licensed under the [MIT License](./LICENSE).
