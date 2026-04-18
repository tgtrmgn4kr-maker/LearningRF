import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.distributions import multinomial
from matplotlib import pyplot as plt

fair_probs = torch.ones([6])

# Sample 500 rolls of a die
counts = multinomial.Multinomial(10 ,probs=fair_probs).sample([500])

cum_counts = counts.cumsum(dim=0)

estimated_probs = cum_counts / cum_counts.sum(dim=1, keepdim=True)

plt.figure(figsize=(6, 4.5))
for  i in range(6):
    plt.plot(estimated_probs[:, i], label=f"Face {i+1}")

plt.xlabel("Number of Rolls")
plt.ylabel("Estimated Probability")
plt.legend()
plt.show()  