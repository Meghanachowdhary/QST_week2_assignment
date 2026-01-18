# Quantum State Reconstruction using Classical Shadows  
### QCG × PaAC Open Project - Assignment 2 (Track 1)

---

## Project Overview

This project explores the problem of **quantum state reconstruction**, where the goal is to recover an unknown quantum **density matrix** using measurement data. Unlike standard machine learning problems, this task comes with strict physical requirements, which makes it both challenging and interesting.

In this work, I use a **Transformer-based machine learning model** under the **Classical Shadows framework** to reconstruct quantum density matrices that are not only accurate, but also physically valid. The project places strong emphasis on enforcing physical constraints, maintaining numerical stability during training, and ensuring that the results can be reproduced reliably.

---

## Objectives

The main goals of this project are:

- To reconstruct a quantum density matrix from given measurement bases and outcomes.
- To ensure that the reconstructed matrix always satisfies key physical properties:
  - Hermiticity  
  - Positive Semi-Definiteness (PSD)  
  - Unit Trace
- To evaluate how close the reconstructed state is to the true state using standard quantum metrics.
- To measure inference latency alongside reconstruction accuracy.

---

## Problem Statement

Given a set of measurement bases and their corresponding outcomes, the task is to reconstruct the underlying density matrix \( \rho \) that fully describes the quantum state.

The main difficulty is that a density matrix cannot be treated like an arbitrary output. It must obey strict mathematical and physical constraints, which are not automatically guaranteed by standard neural network architectures. As a result, these constraints need to be explicitly incorporated into the model design.

---

## Approach

The problem is addressed using a **Transformer-based neural network** following the **Classical Shadows** approach.

The overall workflow is as follows:

- Measurement bases and outcomes are encoded and provided as inputs to the model.
- A Transformer architecture is used to capture relationships across different measurements.
- Instead of predicting the density matrix directly, the model predicts a **lower triangular matrix**.
- This matrix is used to construct a valid density matrix via Cholesky decomposition.
- The output is normalized to ensure the trace equals one.

This design choice allows the model to learn effectively while ensuring that all physical constraints are satisfied by construction.

---

## Physical Constraints and Assumptions

Any valid quantum density matrix must satisfy three essential properties:

- **Hermiticity**
- **Positive Semi-Definiteness**
- **Unit Trace**

To guarantee these properties, the model outputs a lower triangular matrix \( L \), and the density matrix is constructed as:

\[
\rho = \frac{L L^\dagger}{\text{Tr}(L L^\dagger)}
\]

By construction, this formulation ensures that the output always represents a physically valid quantum state, without relying on ad-hoc corrections or penalty terms.

---

## Technologies and Tools

The project was implemented using the following tools and technologies:

- **Programming Language:** Python  
- **Framework:** PyTorch  
- **Libraries:** NumPy, tqdm  
- **Development Environment:** VS Code  
- **Version Control:** Git  

---

## Data Description

- The dataset used in this project is **synthetically generated**.
- Random, physically valid quantum density matrices are sampled.
- Measurement bases and corresponding outcomes are simulated for each state.
- Each data sample contains:
  - Measurement bases
  - Measurement outcomes
  - The ground-truth density matrix

This setup allows the model to be trained in a supervised manner.

---

## Model and Training Details

- **Model Architecture:** Transformer (Track 1 – Classical Shadows)
- **Loss Function:** Frobenius norm squared  
  \[
  \mathcal{L} = \| \rho_{\text{pred}} - \rho_{\text{true}} \|_F^2
  \]
  This loss measures the overall difference between the predicted and true density matrices.
- **Optimizer:** AdamW
- **Key Hyperparameters:**
  - Learning rate: 1e-3
  - Batch size: 32
  - Number of epochs: 20
  - Number of qubits: 2

Special care is taken to avoid numerical instabilities, particularly during trace normalization.

---

## Evaluation Metrics

Model performance is evaluated using the following metrics:

- **Quantum Fidelity**  
  Measures how close the reconstructed quantum state is to the true state.

- **Trace Distance**  
  Measures how distinguishable the predicted state is from the true state.

- **Inference Latency**  
  Measures the time required to reconstruct a density matrix during inference.

---

## Results

A representative evaluation run produced the following results:

- **Mean Fidelity:** 0.9018
- **Mean Trace Distance:** 0.2295 
- **Inference Latency:** approximately  0.72 ms  per reconstruction  

These results indicate that the model is able to reconstruct physically valid quantum states with reasonable accuracy while maintaining low inference latency.

---
## Project Structure

- **src/** – Core source code for the project, including dataset generation, model definition, training, and evaluation.
- **docs/** – Documentation files explaining model design and steps to reproduce results.
- **outputs/** – Saved model checkpoints and evaluation outputs.
- **requirements.txt** – List of Python dependencies required to run the project.
- **AIUSAGE.md** – Disclosure of AI-assisted tools used during the project.
- **README.md** – Project overview, methodology, and usage instructions.

## Conclusion

This project demonstrates that Transformer-based models can be effectively applied
to quantum state reconstruction under the Classical Shadows framework. By enforcing
physical constraints directly within the model design, the approach produces valid
and accurate quantum state reconstructions with low inference latency.







