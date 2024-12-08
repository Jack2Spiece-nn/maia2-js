import numpy as np

# Load logits
logits_python = np.load("maia2_raw_logits.npy")
logits_typescript = np.load("logits_python.npy")
# logits_python_legal = np.load("logits_python_legal.npy")

# with open("logits_typescript.json", "r") as f:
#     import json

#     logits_typescript = np.array(json.load(f))

# with open("logits_typescript_legal.json", "r") as f:
#     logits_typescript_legal = np.array(json.load(f))

# Compute differences
diff = logits_python - logits_typescript
# diff_legal = logits_python_legal - logits_typescript_legal

# Compute statistics
mean_diff = np.mean(diff)
max_diff = np.max(np.abs(diff))
# mean_diff_legal = np.mean(diff_legal)
# max_diff_legal = np.max(np.abs(diff_legal))

print(f"Mean difference (raw logits): {mean_diff}")
print(f"Max difference (raw logits): {max_diff}")
# print(f"Mean difference (legal logits): {mean_diff_legal}")
# print(f"Max difference (legal logits): {max_diff_legal}")

import matplotlib.pyplot as plt

# Plot raw logits
plt.figure(figsize=(12, 6))
plt.plot(logits_python.flatten(), label="Python")
plt.plot(logits_typescript.flatten(), label="TypeScript", alpha=0.7)
plt.title("Raw Logits Comparison")
plt.legend()
plt.show()

# Plot differences
plt.figure(figsize=(12, 6))
plt.plot(diff.flatten())
plt.title("Differences in Raw Logits (Python - TypeScript)")
plt.show()
