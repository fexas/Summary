# The Case of the Lying Simulator: Robustness under Model Misspecification

**"All models are wrong, but some are useful."** — George Box

But what happens when a model is *significantly* wrong? What if the simulator we use to train our AI is overconfident, cleaner, or simpler than the messy reality?

This experiment, **Misspecify Model**, is a stress test. It investigates how different inference methods behave when the "Map" (Simulator) does not match the "Territory" (Reality).

---

## 1. The Scenario: "Precise Simulator, Noisy World"

Imagine we are trying to estimate a parameter $\theta$ (the true location of a signal). 

### The Simulator (The Training World)
We train our neural networks using a simulator. Our simulator believes that measurements are very precise. It generates data points $x$ with a small variance ($\sigma^2_{sim} = 1$).

$$ x_i \sim \mathcal{N}(\theta, 1) $$

### The Reality (The Test World)
In the real world, however, our sensors are actually quite noisy. The real observations $y$ have a much larger variance ($\sigma^2_{obs} = 10$).

$$ y_i \sim \mathcal{N}(\theta_{true}, 10) $$

### The Prior (Our Initial Belief)
Before seeing any data, we believe $\theta$ follows a standard normal distribution (scaled):
$$ \theta \sim \mathcal{N}(0, 25) $$

---

## 2. The Conflict: Out-Of-Distribution (OOD)

This setup creates a fundamental conflict called **Model Misspecification**.

1.  **Training Phase**: The AI sees thousands of examples where the data scatter (variance) is always $\approx 1$. It learns rules based on this fact.
2.  **Inference Phase**: We show the AI a real observation where the data scatter is $\approx 10$.

This is an **Out-Of-Distribution** input. The AI has never seen data this noisy. It's like training a self-driving car only on sunny days and then asking it to drive in a blizzard.

---

## 3. The Contenders

We compare two methods to see which one breaks first.

### Contender A: NPE (Neural Posterior Estimation)
*   **Input**: We manually calculate summary statistics: Sample Mean ($\bar{x}$) and Sample Variance ($s^2$).
*   **The Logic**: NPE learns the probability distribution $p(\theta | \bar{x}, s^2)$.
*   **The Failure Mode**: During training, NPE *always* sees $s^2 \approx 1$. It essentially learns to rely on this. When we feed it a real observation with $s^2 \approx 10$, the network is confused. It tries to extrapolate its learned rules to this alien value. Often, this results in **overconfidence** (a posterior that is too narrow) or **bias** (the peak is in the wrong place), because the network cannot reconcile the high variance with its training data.

### Contender B: SMMD (Sliced Maximum Mean Discrepancy)
*   **Input**: The raw set of data points $\{x_1, x_2, ..., x_{100}\}$.
*   **The Logic**: SMMD trains a summary network to extract relevant features automatically, optimizing the Sliced-MMD distance between the simulator and a generator.
*   **The Hope**: By learning from raw data, does SMMD learn a more robust representation? Or does it also collapse? We test if learning summaries end-to-end offers any protection against this type of misspecification.

---

## 4. The Mathematical Breakdown

Let's look at the exact distributions involved.

**1. The True Posterior (The Ground Truth)**
Since we know the *real* variance is 10, we can calculate the mathematically correct answer (the Green curve in our plots).
$$ p(\theta | y) \propto p(y | \theta) p(\theta) $$
Because the noise is high ($\sigma^2=10$), the True Posterior should be **wide** (indicating high uncertainty).

**2. The Misspecified "Simulator" Posterior**
If we naively trusted our simulator (which thinks $\sigma^2=1$), we would calculate a posterior that is **very narrow** (highly confident). This is the trap.

---

## 5. The Results (What actually happened?)

When you view the generated plot (`misspecification_results.png`), you will see:

*   **Green Curve (True Posterior)**: It is wide and covers the true parameter value. This represents the correct level of uncertainty given the noisy data.
*   **Blue Curve (NPE)**: It often fails dramatically. It might place its probability mass far away from the true value or be oddly shaped. This is because the input "Variance=10" acts as an adversarial example to a network that only knows "Variance=1".
*   **Orange Curve (SMMD)**: This shows how our method behaves. Ideally, we want it to be closer to the Green curve (robust) rather than failing like the Blue curve.

## 6. Key Takeaway

**" Garbage In, Garbage Out" applies to Model Assumptions too.**

If your simulator assumes the world is clean, your AI will be brittle in a messy reality. Standard inference methods (like NPE on fixed summaries) are highly sensitive to this. This experiment demonstrates why we need **Robust Inference** methods—methods that don't panic when the map doesn't perfectly match the territory.
