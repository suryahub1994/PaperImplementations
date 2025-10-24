# Paper Implementations (Barebones Edition)

This repository contains **from-scratch implementations** of classical and modern algorithms or architectures described in various research papers.

The guiding idea is to **replicate the logic** of each paper in a way that is **accessible to software developers**, focusing on *clarity over cleverness*.

---

## 💡 Philosophy

- **C++ is preferred over Python**
  - When performance or control over memory is essential.
- **Python is preferred over heavy libraries**
  - Use it for readability and quick experimentation, but avoid frameworks that obscure the core algorithm.
- **Non-vectorized code is preferred over vectorized code**
  - To expose the *actual computational flow* behind each algorithm.
  - Each loop and operation is explicit — no hidden magic from NumPy, PyTorch, etc.

> The goal: *to understand how things work under the hood, not just make them work.*

---

## 🧩 Structure

Each folder corresponds to a specific paper or technique.  
For each, you’ll typically find:
- Paper
- Code

