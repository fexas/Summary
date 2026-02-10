# A.3 Lotka-Volterra population model

The Lotka–Volterra model [84] is a Markov jump process describing the evolution of a population of predators interacting with a population of prey, and has four parameters $\theta = (\theta_1, \dots, \theta_4)$. Let $X$ be the number of predators, and $Y$ be the number of prey. According to the model, the following can take place:

*   With rate $\exp(\theta_1)XY$ a predator may be born, increasing $X$ by one.
*   With rate $\exp(\theta_2)X$ a predator may die, decreasing $X$ by one.
*   With rate $\exp(\theta_3)Y$ a prey may be born, increasing $Y$ by one.
*   With rate $\exp(\theta_4)XY$ a prey may be eaten by a predator, decreasing $Y$ by one.

Our experimental setup follows that of Papamakarios and Murray [59]. We used initial populations $X = 50$ and $Y = 100$. We simulated the model using the Gillespie algorithm [24] for a total of 30 time units. We recorded the two populations every 0.2 time units, which gives two timeseries of 151 values each. The data $x$ is 9-dimensional, and corresponds to the following timeseries features:

*   The mean of each timeseries.
*   The log variance of each timeseries.
*   The autocorrelation coefficient of each timeseries at lags 0.2 and 0.4 time units.
*   The cross-correlation coefficient between the two timeseries.

Each feature was normalized to have approximately zero mean and unit variance based on a pilot run. The ground truth parameters were taken to be:

$$\theta^*=(\log 0.01,\log 0.5,\log 1,\log 0.01),\qquad(25)$$

and the observed data $\mathbf{x}_o$ were generated from a simulation of the model at $\mathbf{\theta}^*$. In our experiments we used two priors: (a) a broad prior defined by:

$$p_{\text{broad}}(\theta)\propto\prod_{i=1}^{4}I(-5\leq\theta_{i}\leq 2),\qquad(26)$$

and (b) a prior corresponding to the oscillating regime, defined by:

$$p_{\text{osc}}(\theta)\propto\mathcal{N}(\theta|\theta^{*},0.5^{2})p_{\text{broad}}(\theta).\qquad(27)$$
