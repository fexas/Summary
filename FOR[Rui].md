# Project Overview: Simulation-Based Inference with SMMD
**Welcome, Rui!** This document is your "hitchhiker's guide" to the project we've been building. It's designed to give you the high-level context, the "why" behind our code, and the lessons learned from the trenches.

---

## 1. The Big Picture: What are we doing?

Imagine you have a complex computer simulation (like a weather model or a stock market simulator). You can run it forward easily: *Input parameters $\to$ Output data*.

But often, we want to do the reverse: *Observe real data $\to$ Figure out the parameters that caused it*. This is **Inverse Problem** or **Parameter Inference**.

In many real-world cases, we **cannot write down the probability formula (likelihood)** for these simulators. This is where **Simulation-Based Inference (SBI)** comes in. We use neural networks to learn this relationship.

### The Problem with Traditional SBI
Traditional methods (like NPE) often struggle when:
1.  **Data is high-dimensional**: Processing raw images or time-series is hard.
2.  **Summary statistics are manual**: We usually guess what features matter (mean? variance?), losing information.
3.  **The model is "Wrong" (Misspecification)**: The simulator is never a perfect copy of reality. If the simulator says "variance is 1" but reality says "variance is 10", standard methods can become confidently wrong.

### Our Solution: SMMD (Sliced Maximum Mean Discrepancy)
We are testing a method called **SMMD**. Instead of manually picking summary statistics, we train a neural network to **learn** the best summary statistics automatically. We do this by minimizing a distance metric called **Sliced MMD** between our simulator's output and a generator's output.

---

## 2. The Three Main Experiments

We designed three "arenas" to test our gladiator (SMMD) against the others.

### Arena 1: The Classic Benchmark (G-and-k Distribution)
*   **Location**: `G_and_K/`
*   **The Task**: Infer 4 parameters ($A, B, g, k$) of a statistical distribution that has no closed-form density.
*   **The Goal**: Test how well SMMD performs under different "beliefs" (Priors).
    *   **Informative Prior**: We have a good guess where the parameters are.
    *   **Vague/Weak Prior**: We have no clue.
*   **Outcome**: We showed SMMD is robust even when our prior knowledge is weak, unlike some baselines that might struggle to converge.

### Arena 2: The Dimension & Distance Test (Multivariate Gaussian)
*   **Location**: `Multivariate_Gaussian/`
*   **The Task**: Infer the variance of a high-dimensional Gaussian.
*   **The Goal**: Compare SMMD against **ABC (Approximate Bayesian Computation)** methods using different distance metrics.
    *   **Euclidean**: Basic distance (fails in high dims).
    *   **Sliced Wasserstein (SW)**: A fancy Optimal Transport metric.
    *   **Hilbert Swap**: A clever way to sort high-dim data using space-filling curves.
*   **Key Insight**: We split "Hilbert Swap" into two distinct metrics to see exactly where the gain comes from. SMMD learns summary stats that make simple Euclidean distance work as well as the fancy metrics!

### Arena 3: The "Stress Test" (Model Misspecification)
*   **Location**: `misspecify_model/`
*   **The Task**: The simulator generates data with variance=1. The "Real World" (Observation) has variance=10.
*   **The Goal**: See who breaks.
    *   **NPE (Neural Posterior Estimation)**: It sees the "variance" statistic is 10. Since it was trained only on variance=1, it panics (or rather, confidently outputs garbage).
    *   **SMMD**: Trained on raw data. We wanted to see if learning summaries from scratch makes it more robust.
*   **Outcome**: The plot in `misspecify_model/misspecification_results.png` shows NPE failing (blue curve in the wrong place), while the True posterior (green) is broad.

---

## 3. The Codebase: A Map of the Territory

*   **`models.py` (in `Multivariate_Gaussian/`)**: The brain. Contains the `SMMD_Model`, `SummaryNet`, and `InvariantModule`. We use "Deep Sets" architecture (Invariant layers) so the order of data points doesn't matter.
*   **`experiment*.py`**: The runners. These scripts set up the simulator, train the models, and plot results.
*   **`rnpe-main/`**: The reference library. We looked at this JAX code but re-implemented the logic in PyTorch to keep our stack consistent.

---

## 4. Technical Decisions & Tech Stack

*   **PyTorch vs JAX**: The reference paper used JAX. We chose **PyTorch** because:
    *   It's what we are most comfortable with.
    *   Integration with `sbi` package is native.
    *   Debugging is easier (eager execution).
*   **SBI Package**: We used the standard `sbi` library for the NPE baseline to ensure a fair comparison against "industry standard".
*   **Optimal Transport (POT)**: Used for Wasserstein distances in the ABC experiments.

---

## 5. Lessons Learned (The "War Stories")

Here is what being an engineer on this project taught us:

### 1. The "Apple Silicon" Trap (MPS vs CPU)
**The Bug**: We tried running neural networks on Mac's GPU (MPS). It crashed with vague errors or produced NaNs when interacting with `sbi` or `arviz`.
**The Fix**: **Simplicity first**. We forced `device='cpu'` for these specific experiments.
**Lesson**: Don't blindly optimize for speed. Get it working on CPU first. If it takes 2 minutes instead of 30 seconds, that's fine for prototyping.

### 2. The "Arviz Permission" Ghost
**The Bug**: Running the code caused a `PermissionError` trying to write to a hidden folder in the home directory.
**The Fix**: We set environment variables `os.environ['ARVIZ_DATA'] = '/tmp'` inside the script.
**Lesson**: Libraries often have hidden side effects (like writing config files). When you see a permission error, check environment variables.

### 3. The "OOD" Trap (Out-Of-Distribution)
**The Insight**: In the Misspecification experiment, NPE failed not because it's "bad", but because the input summary statistic (variance=10) was completely outside the range it saw during training (variance $\approx$ 1).
**Lesson**: Neural networks extrapolate poorly. If your real data looks different from your training data, trust nothing.

### 4. Vectorization is King
**The Insight**: In the Gaussian simulator, generating data point-by-point is slow. We used `numpy` broadcasting to generate 10,000 samples at once.
**Lesson**: Always ask, "Can I do this operation on the whole array at once?"

---

## 6. How to "Think Like a Good Engineer"

1.  **Modularize**: We separated `models.py` so we could reuse the SMMD architecture across different experiments. Don't copy-paste code; import it.
2.  **Visual Verification**: We didn't just look at loss numbers. We plotted the posteriors (kde plots). If the loss is low but the plot looks wrong, the model is wrong.
3.  **Reproducibility**: We set random seeds. We defined config constants (`THETA_DIM`, `SUMMARY_DIM`) at the top of files instead of burying magic numbers in the code.
4.  **Bias for Action**: When we needed to compare against the `rnpe` paper, we didn't wait to learn JAX perfectly. We read the logic and ported it to PyTorch. **Language is syntax; logic is universal.**

---

*Keep this guide handy. It is the map to the world we built.*
