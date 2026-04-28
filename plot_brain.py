import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results_brain/brain_results.csv")
os.makedirs("results_brain/plots", exist_ok=True)

# 1. Freeze vs Accuracy
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, model in enumerate(["resnet50", "efficientnet_b3"]):
    ax = axes[i]
    subset = df[(df.model == model) & (df.data_split == 1.0)].sort_values("freeze_fraction")
    ax.plot(subset.freeze_fraction * 100, subset.best_val_acc, marker="o")
    ax.set_title(f"{model} — Brain Tumor")
    ax.set_xlabel("Layers frozen (%)")
    ax.set_ylabel("Best val accuracy")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results_brain/plots/brain_freeze_vs_acc.png", dpi=150)
plt.close()

# 2. Model Comparison
fig, ax = plt.subplots(figsize=(8, 5))
for model in ["resnet50", "efficientnet_b3"]:
    subset = df[(df.model == model) & (df.data_split == 1.0)].sort_values("freeze_fraction")
    ax.plot(subset.freeze_fraction * 100, subset.best_val_acc, marker="o", label=model)
ax.set_title("ResNet-50 vs EfficientNet-B3 — Brain Tumor")
ax.set_xlabel("Layers frozen (%)")
ax.set_ylabel("Best val accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results_brain/plots/brain_model_comparison.png", dpi=150)
plt.close()

# 3. Data Split Effect
fig, ax = plt.subplots(figsize=(8, 5))
for model in ["resnet50", "efficientnet_b3"]:
    subset = df[(df.model == model) & (df.freeze_fraction == 0.0)].sort_values("data_split")
    ax.plot(subset.data_split * 100, subset.best_val_acc, marker="o", label=model)
ax.set_title("Data Efficiency — Brain Tumor")
ax.set_xlabel("Training data (%)")
ax.set_ylabel("Best val accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results_brain/plots/brain_data_split.png", dpi=150)
plt.close()

# 4. Training Time
fig, ax = plt.subplots(figsize=(8, 5))
for model in ["resnet50", "efficientnet_b3"]:
    subset = df[(df.model == model) & (df.data_split == 1.0)].sort_values("freeze_fraction")
    ax.plot(subset.freeze_fraction * 100, subset.wall_time_seconds / 60, marker="o", label=model)
ax.set_title("Training Time vs Freeze Level — Brain Tumor")
ax.set_xlabel("Layers frozen (%)")
ax.set_ylabel("Training time (minutes)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results_brain/plots/brain_training_time.png", dpi=150)
plt.close()

print("All brain tumor plots saved!")