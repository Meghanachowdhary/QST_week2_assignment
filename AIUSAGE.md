# AI Usage Disclosure

This project was completed with limited support from AI-assisted tools. The use of
AI is disclosed here to maintain transparency and follow the project’s AI
attribution guidelines.

---

## How AI Was Used

During the development of this project, I used **ChatGPT** as a **learning and
support tool** at different stages.

AI assistance was mainly used when:
- I was unsure about theoretical concepts related to quantum state tomography
- I faced implementation or debugging issues in Python and PyTorch
- I needed help understanding errors or improving code structure
- I was writing and organizing documentation files such as the README and
  replication guide

ChatGPT was used in a similar way to an online reference or discussion partner.
It helped me understand ideas, but I always made sure I understood the solution
before using or modifying it.

All important decisions — including model design, mathematical formulation,
training strategy, and evaluation — were made by me.

No other AI tools (such as Claude or GitHub Copilot) were used in this project.

---

## Example Prompts Used

Below are examples of questions I asked ChatGPT naturally while working on this
project:

- *I need code to generate random valid density matrices for training. Can you
  show me a simple Python example?*

- *My model outputs a matrix. How can I convert it into a valid density matrix
  with trace equal to 1?*

- *Can you give a PyTorch example where the model outputs a lower triangular
  matrix and then constructs a valid density matrix from it?*

- *I’m getting an error saying MSE loss is not implemented for complex tensors.
  What loss should I use instead?*

- *My training loss is becoming NaN. Can you help me understand what might be
  going wrong?*

- *Can you help me clean up the indentation and structure of my training loop?*

- *I’m confused about how to properly run this project. What command should I
  use to run `train.py`?*

- *Why am I getting a “No module named src” error, and how do I fix my imports?*

- *How do I create and activate a virtual environment in VS Code?*

- *What Python libraries do I need to install to run this project without
  errors?*

- *Can you help me write a simple AI usage disclosure file?*

These prompts reflect the typical way AI was used — mainly for understanding,
debugging, and documentation.

---

## Verification and Responsibility

All AI-assisted suggestions were carefully checked before being included in the
project.

I verified the work by:
- Manually checking mathematical expressions and logic
- Ensuring predicted density matrices satisfy physical constraints such as
  Hermiticity, positive semi-definiteness, and unit trace
- Testing code outputs and modifying AI-suggested snippets when needed
- Validating results using quantum fidelity and trace distance metrics
- Checking for numerical stability during training and evaluation

Only content that I fully understood and confirmed to be correct was included in
the final submission.

---

## Declaration

I confirm that AI tools were used responsibly and transparently, and that this
project represents my own understanding, implementation, and verification of the
methods used.
