import os
import numpy as np
import torch
import pyabc
import ot
import scipy.stats as stats
import logging

logging.getLogger("pyabc").setLevel(logging.WARNING)

class W2Distance(pyabc.Distance):
    def __call__(self, x, y, t=None, par=None):
        X = np.asarray(x["data"])
        Y = np.asarray(y["data"])
        n = X.shape[0]
        m = Y.shape[0]
        a = np.ones((n,)) / n
        b = np.ones((m,)) / m
        M = ot.dist(X, Y, metric="sqeuclidean")
        w2_sq = ot.emd2(a, b, M)
        return np.sqrt(w2_sq)

def run_smc_abc(task, x_obs, n_samples=1000, max_populations=10, 
                result_dir="results", distance_metric="w2"):
    prior_dict = {}
    for i in range(task.d):
        loc = task.lower[i]
        scale = task.upper[i] - task.lower[i]
        prior_dict[f"theta{i}"] = pyabc.RV("uniform", loc, scale)
    prior = pyabc.Distribution(**prior_dict)

    def model_wrapper(params):
        theta = np.zeros(task.d)
        for i in range(task.d):
            theta[i] = params[f"theta{i}"]
        x_sim = task.simulator(theta[np.newaxis, :])[0]
        return {"data": x_sim}

    if distance_metric == "w2":
        distance = W2Distance()
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    abc = pyabc.ABCSMC(
        models=model_wrapper,
        parameter_priors=prior,
        distance_function=distance,
        population_size=n_samples,
        sampler=pyabc.sampler.SingleCoreSampler(),
    )

    if x_obs.ndim == 3:
        x_obs_data = x_obs[0]
    else:
        x_obs_data = x_obs

    os.makedirs(result_dir, exist_ok=True)
    db_name = f"{distance_metric}abc_temp.db"
    db_path = os.path.join(result_dir, db_name)
    if os.path.exists(db_path):
        os.remove(db_path)
    db_url = "sqlite:///" + db_path
    abc.new(db_url, {"data": x_obs_data})
    history = abc.run(max_nr_populations=max_populations)
    df, w = history.get_distribution(m=0, t=history.max_t)
    samples = np.zeros((len(df), task.d))
    for i in range(task.d):
        samples[:, i] = df[f"theta{i}"].values
    indices = np.random.choice(len(df), size=n_samples, p=w / w.sum(), replace=True)
    final_samples = samples[indices]
    return final_samples

def run_w2abc(task, x_obs, n_samples=1000, max_populations=10, result_dir="results"):
    return run_smc_abc(task, x_obs, n_samples, max_populations, result_dir, distance_metric="w2")

