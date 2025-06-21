import time

import matplotlib.pyplot as plt
import numpy as np
import torch


# --- Resampling methods ---
def multinomial_resample(weights):
    return torch.multinomial(weights, len(weights), replacement=True)

def systematic_resample(weights):
    N = len(weights)
    positions = (torch.rand(1) + torch.arange(N)) / N
    cumsum = torch.cumsum(weights, dim=0)
    return torch.searchsorted(cumsum, positions)

def stratified_resample(weights):
    N = len(weights)
    positions = (torch.rand(N) + torch.arange(N)) / N
    cumsum = torch.cumsum(weights, dim=0)
    return torch.searchsorted(cumsum, positions)

def residual_resample(weights):
    N = len(weights)
    indexes = []
    num_copies = (N * weights).int()
    for i in range(N):
        indexes.extend([i] * num_copies[i])
    remaining = N - len(indexes)
    if remaining > 0:
        residual = weights - num_copies.float() / N
        residual /= residual.sum()
        indexes += torch.multinomial(residual, remaining, replacement=True).tolist()
    return torch.tensor(indexes)

# --- Configuration ---
resamplers = {
    "Multinomial": multinomial_resample,
    "Systematic": systematic_resample,
    "Stratified": stratified_resample,
    "Residual": residual_resample
}

N = 1000
weights = torch.rand(N)
weights = weights / weights.sum()

# --- Execution ---
results = {}
for name, method in resamplers.items():
    start = time.time()
    idxs = method(weights)
    duration = time.time() - start
    unique, counts = np.unique(idxs.numpy(), return_counts=True)
    var = np.var(counts)
    results[name] = {
        "czas [s]": duration,
        "wariancja powtórzeń": var,
        "counts": counts
    }

# --- Plot histograms ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()
for i, (name, data) in enumerate(results.items()):
    axs[i].hist(data["counts"], bins=30, color="skyblue", edgecolor="black")
    axs[i].set_title(f"{name}\nVar: {data['wariancja powtórzeń']:.2f}, Time: {data['czas [s]']:.4f}s")
    axs[i].set_xlabel("Liczba powtórzeń")
    axs[i].set_ylabel("Liczba cząstek")

plt.tight_layout()
plt.show()
