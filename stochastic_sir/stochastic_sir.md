## Example 2. The stochastic SIR model

Consider three classes of individuals: **susceptible**, **infected**, and **removed** (by recovery or death). 

As in the previous example, we will use $S$, $I$ and $R$ to represent the compartments themselves, as well as the numbers of individuals in each compartment, and assume $S + I + R = N$, a constant.

### Transition Probabilities

Thus, in the time interval $[t, t + \Delta t]$, the probability of an infection, that is, the simultaneous transitions $S \longrightarrow S-1$ and $I \longrightarrow I+1$, is:
$$\beta \frac{SI}{N}\Delta t + o(\Delta t)$$
If it is assumed that infected individuals recover with rate $\gamma$, the probability for a recovery, $I \longrightarrow I-1$ and $R \longrightarrow R+1$, in the interval $[t, t + \Delta t]$, is:
$$\gamma I \Delta t + o(\Delta t)$$
Because $R = N - S - I$, it is enough to consider the process $($S_t$, $I_t$)$. The probabilities of an infection and of a recovery during the time interval $[t, t + \Delta t]$ are:
$$

\begin{aligned}
P\left((S_{t+\Delta t}, I_{t+\Delta t}) - (S_t, I_t) = (-1, 1)\right) &= \beta \frac{S_t I_t}{N}\Delta t + o(\Delta t), \\
P\left((S_{t+\Delta t}, I_{t+\Delta t}) - (S_t, I_t) = (0, -1)\right) &= \gamma I_t \Delta t + o(\Delta t),
\end{aligned}
$$

$$
$$
with the complementary probability:
$$
P\left((S_{t+\Delta t}, I_{t+\Delta t}) - (S_t, I_t) = (0, 0)\right) = 1 - \left(\beta \frac{S_t}{N} + \gamma\right)I_t \Delta t + o(\Delta t).
$$
This model, widely known as the **general stochastic epidemic**, was introduced by Bartlett in 1949.

### Stochastic Equations

The stochastic equations describing this process are obtained by adding and subtracting, to each increment of $S_t$ and $I_t$, the conditional expectations, given the value of the process at the beginning of the corresponding time increment, say, of length $\Delta t$.

Each increment of the process can be expressed as the expected value of the increment plus a sum of centered increments. In our example, the expected values of the increments $\Delta S = S_{t+\Delta t} - S_t$ and $\Delta I = I_{t+\Delta t} - I_t$ are:
$$-\beta \frac{S_t I_t}{N}\Delta t \quad \text{and} \quad \left(\beta \frac{S_t I_t}{N} - \gamma I_t\right)\Delta t$$
respectively, so the increments can be written as:
$$

\begin{aligned}
\Delta S &= \left(-\beta \frac{S_t I_t}{N}\right)\Delta t + \Delta Z_1 \\
\Delta I &= \left(\beta \frac{S_t I_t}{N} - \gamma I_t\right)\Delta t - \Delta Z_1 + \Delta Z_2,
\end{aligned}
$$

$$
$$
where $\Delta Z_1$ and $\Delta Z_2$ are conditionally centered Poisson increments with mean zero and conditional variances $\beta(S_t I_t/N)\Delta t$ and $\gamma I_t \Delta t$.

### Deterministic Limit (ODE)

Now let us consider what happens if we drop the terms $\Delta Z_i$ from equations above, and let $\Delta t \to 0$. The resulting ordinary differential equations define a deterministic model:
$$
\begin{aligned}
\frac{dS}{dt} &= -\beta \frac{S_t I_t}{N}, \\
\frac{dI}{dt} &= \beta \frac{S_t I_t}{N} - \gamma I_t,
\end{aligned}
$$

$$
If $\hat{\beta} S_t I_t$ is used instead of $\beta S_t I_t/N$, with $\hat{\beta} = \beta/N$, we have, after dropping the hats, the so called **Kermack and McKendrick ODE model**:
$$

\begin{aligned}
\frac{dS}{dt} &= -\beta S_t I_t, \\
\frac{dI}{dt} &= \beta S_t I_t - \gamma I_t.
\end{aligned}

$$
