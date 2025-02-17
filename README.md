# In-Context Learning and Preconditioned Gradient Descent in Transformers

## Overview

This repository contains the code for numerical experiments accompanying the research paper **"Decoding In-Context Learning: The Role of Preconditioned Gradient Descent in Transformers"**, which is included in this repository as **`Decoding_In-Context_Learning.pdf`**. The code is also part of a broader ongoing research project in this area.

The paper investigates how **Transformers can mimic Preconditioned Gradient Descent (PGD) in in-context learning**, with both theoretical analysis and empirical experiments. The methodology is explained in the paper. My findings highlight the similarities and differences between in-context learning and classical optimization methods.

The code in this repository is largely based on the theory from:
- ["Transformers learn in-context by gradient descent" (von Oswald et al., 2023)](https://arxiv.org/abs/2212.07677)
- ["Transformers learn to implement preconditioned gradient descent" (Ahn et al., 2023)](https://arxiv.org/abs/2306.00297)

## Running Experiments

The **`runner.py`** file provides an example of how to set up and execute experiments. The core experiment logic is implemented in **`run_experiments.py`**. The repository allows testing **different Transformer architectures** (Linear, ReLU, LeakyReLU, Softmax) and comparing their in-context learning performance to PGD baselines.

### Running an Experiment:

Execute `runner.py` to start an experiment:
This script:
- Generates **tokenized linear regression data**
- Trains **various Transformer models**
- Evaluates **performance against PGD baselines**
- Logs **metrics and saves results** in `results/`

## Experiment Analysis Tools

Beyond training, the repository also provides **analysis tools** to process and visualize results:

- **`convert_and_plot.py`**  
  Converts `.pt` log files to `.csv` and generates **plots comparing different models**.
- **`experiment_summary.py`**  
  Aggregates and summarizes results across multiple experiments.  
  It crawls experiment folders, extracts key metrics from CSV logs, and creates a **summary table**.  
  The table is formatted for **LaTeX export** to include in reports.
