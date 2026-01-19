# Model Working

This project follows the **Track 1** approach and focuses on reconstructing a
quantum density matrix from classical shadow measurement data using a deep
learning model. The goal is to learn a reliable mapping from classical
measurement outcomes to a physically valid quantum state representation.

---

## Overall Idea

Classical shadow measurements provide partial and noisy information about a
quantum state. A single measurement does not fully describe the state. However,
when many such measurements are combined, it becomes possible to reconstruct
the underlying density matrix.

In this project, a Transformer based neural network is used to learn this
reconstruction process directly from data.

---

## Model Architecture

A **Transformer encoder** is used as the core model architecture. The
Transformer is well suited for this task because it can process long input
sequences and learn relationships between distant elements in the data.

This is important for classical shadow data, where useful information may be
distributed across many measurements obtained from different qubits and
shadows.

The self attention mechanism allows the model to assign different importance
to different measurements and combine them effectively to form a global
representation of the quantum state.

---

## Input Representation

The input to the model consists of measurement data obtained from the
Classical Shadows protocol. Each measurement includes:

- A measurement basis
- A corresponding measurement outcome

These values are first converted into discrete tokens. Tokenization provides a
structured way to represent measurement events. The tokens are then mapped to
embedding vectors, converting discrete inputs into continuous representations
that can be processed by the Transformer.

---

## Transformer Processing

The embedded measurement sequence is passed through multiple Transformer
encoder layers. Using self attention, the model learns how different
measurements are related to each other and how they jointly contribute to the
reconstruction of the quantum state.

Although classical shadow measurements are theoretically order-invariant, the
measurements are treated as a sequence for simplicity and ease of
implementation.

---

## Output Representation

Instead of predicting the density matrix directly, the model outputs a vector
that represents a **lower triangular matrix** \( L \).

- Diagonal elements of \( L \) are constrained to be real
- Off-diagonal elements may be complex

This intermediate representation is later used to construct a valid density
matrix.

---

## Enforcing Physical Constraints

To ensure physical validity, the density matrix is constructed using:

The density matrix is obtained by multiplying the lower triangular matrix `L` with its conjugate transpose and normalizing the result so that the trace equals one.


This construction guarantees:

- **Hermiticity**
- **Positive Semi-Definiteness**
- **Unit Trace**

By enforcing these constraints directly through the model design, the network
cannot produce physically invalid quantum states during training or inference.

---

## Training Procedure

The model is trained using the **Frobenius norm** as the loss function. This
loss measures the element wise difference between the predicted and true
density matrices.

Gradient-based optimization is used to minimize this loss over the training
dataset, encouraging accurate reconstruction of quantum states.

---

## Evaluation Metrics

Model performance is evaluated using the following metrics:

- **Quantum Fidelity**, which measures how close the predicted quantum state is to the true state.
- **Trace Distance**, which measures how distinguishable the predicted and true quantum states are.
- **Inference Latency**, which measures the time required to reconstruct a single quantum state during inference.

Together, these metrics provide a clear assessment of both the reconstruction accuracy and the computational efficiency of the model.


## Summary

This project combines classical shadow measurements with a Transformer based
deep learning model to perform quantum state reconstruction. By carefully
designing the input representation, model architecture, and output constraints,
the approach ensures both accurate reconstruction and strict physical validity
of the predicted quantum states.


