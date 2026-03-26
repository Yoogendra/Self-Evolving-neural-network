import matplotlib.pyplot as plt
import numpy as np

# --- INSERT YOUR ACTUAL RECORDED NUMBERS HERE ---
baseline_acc = 71.88       # Replace with your train_baseline.py result
baseline_params = 2.12     # Replace with your train_baseline.py result (in Millions)

senn_acc = 81.20           # Replace with final_metrics.json result
senn_params = 0.16        # Replace with final_metrics.json result (in Millions)
# ------------------------------------------------

models = ['Standard Baseline CNN', 'SENN (Evolved)']
accuracies = [baseline_acc, senn_acc]
parameters = [baseline_params, senn_params]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Accuracy (Higher is better)
bars1 = ax1.bar(models, accuracies, color=['#95a5a6', '#2ecc71'])
ax1.set_title('CIFAR-10 Classification Accuracy', fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 100)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', fontweight='bold')

# Plot 2: Efficiency (Lower is better)
bars2 = ax2.bar(models, parameters, color=['#e74c3c', '#3498db'])
ax2.set_title('Model Size (Parameter Count)', fontweight='bold')
ax2.set_ylabel('Parameters (Millions)')

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}M", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('SENN_Evaluation_Panel.png', dpi=300)
print("Graph saved as SENN_Evaluation_Panel.png!")