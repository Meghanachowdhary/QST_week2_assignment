# Replication Guide

This document explains how the results of this project can be reproduced. It
covers the environment setup, dataset generation, training, and evaluation.

---

## Step 1: Environment Setup

The project was developed using Python (version 3.8 or later).

A virtual environment was used to manage dependencies. All required Python
packages are listed in the `requirements.txt` file. Installing these
dependencies ensures the project runs correctly.

---

## Step 2: Dataset Generation

The dataset is generated within the project and does not rely on external data.

Random quantum density matrices and corresponding measurement data are generated
using the Classical Shadows method. This logic is implemented in `src/data.py`.

Dataset generation happens automatically during training.

---

## Step 3: Model Training

The model is trained using a Transformer-based architecture implemented in
`src/train.py`.

During training, the model learns to reconstruct density matrices from
measurement data. Training progress and loss values are printed to the terminal.

The trained model is saved in the `outputs` directory.

---

## Step 4: Evaluation

After training, the saved model is evaluated using the evaluation script.

Evaluation reports quantum fidelity, trace distance, and inference latency. These
metrics are used to measure reconstruction accuracy and performance.

---

## Summary

By following the steps above, the full experimental setup of this project can be
reproduced. Small variations in results may occur due to randomness.
