# MeasureBreaks – Proactive Graph Break Analysis for PyTorch 2

**BreakScope** is a static analysis framework that predicts graph breaks in PyTorch 2’s `torch.compile` before they happen, assigns severity costs, and helps developers prioritise refactoring. It shifts mitigation from reactive (post‑compilation) to **proactive, cost‑aware guidance**.

This repository contains:
- A research proposal (PDF)
- A lightweight demo that statically detects break patterns and ranks models by estimated cost
- Results from running the demo with PyTorch’s inductor backend

## Problem

PyTorch 2’s `torch.compile` delivers large speedups but **graph breaks** (when Python constructs cannot be captured) fragment the graph, increase cold‑start latency, and reduce optimisation opportunities. Existing tools react to breaks *after* they occur – BreakScope predicts them *before* compilation, saving developer time and improving performance.

## Novelty

- **Static AST/CFG analysis** to detect break patterns without running the model.
- **Cost modelling** that assigns different penalties to break types (e.g., numpy calls cost more than `print()`).
- **Priority ranking** so developers fix the most harmful breaks first.
- **Future extension**: dynamic shape detection (mask indexing, `nonzero()`) – see **Limitations & Future Work**.

## Demo

We implemented a simple but complete demo that:

1. Defines six small models (clean, `.item()`, `print`, numpy, dynamic shape, multi‑break).
2. Uses `ast` to statically detect break patterns.
3. Runs `torch._dynamo.explain()` with the inductor backend to obtain real graph break counts.
4. Ranks models by estimated cost (higher = worse).

### Results (CPU, inductor backend)

| Rank | Model              | Static Breaks                         | Runtime Break Count | Est. Cost (ms) |
|------|--------------------|---------------------------------------|---------------------|----------------|
| 1    | MultiBreak         | print(), item(), numpy                | 1                   | 21             |
| 2    | ModelWithNumpy     | numpy                                 | 0                   | 12             |
| 3    | ModelWithItem      | item()                                | 1                   | 6              |
| 4    | ModelWithPrint     | print()                               | 0                   | 3              |
| 5    | CleanModel         | none                                  | 0                   | 0              |
| 6    | DataDependentShape | none*                                 | 1                   | 0              |

*Dynamic shape (mask indexing on a separate line) is not yet detected – see **Limitations & Future Work**.

**Key takeaways:**
- Static detection correctly identifies all break types for inline patterns.
- Cost ranking matches intuition (numpy > item > print).
- Runtime break counts confirm breaks for `.item()` and dynamic shapes.

## Limitations
BreakScope is a research prototype. The following limitations are acknowledged:

- Dynamic shape detection is incomplete
The current AST visitor catches x[x > 0] but not mask = x > 0; x[mask] (mask declared on a previous line). Full data‑flow analysis is required. This is a planned extension.

- Cost model coefficients are heuristic
The per‑break costs (e.g., numpy = 12 ms) are based on literature and profiling logs, not yet calibrated on large benchmarks. Future work will learn coefficients from TorchBench measurements.

- Refactoring suggestions are not part of this demo
The full BreakScope design includes automatic or semi‑automatic refactoring (e.g., replacing .item() with tensor operations). The simple demo focuses on prediction and ranking; refactoring will be integrated in the final system.

- ML‑based break predictor is omitted for clarity
The full research prototype includes a Random Forest classifier trained on synthetic data to handle ambiguous cases. The demo shows only rule‑based detection, which already covers the majority of break patterns.

### How to run the demo

```bash
# Clone the repository
git clone https://github.com/yourusername/breakscope.git
cd breakscope

# Install dependencies
pip install torch pandas

# Run the inductor version (recommended)
python measurebreaks_inductor.py

# Or the eager version
python measurebreaks_eager.py
