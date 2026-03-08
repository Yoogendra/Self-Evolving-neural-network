# 🚀 SENN – Self-Evolving Neural Network **Automatic CNN Architecture Search via Evolutionary Intelligence**

## 📌 Overview

**SENN (Self-Evolving Neural Network)** is a research-grade evolutionary framework that **automatically discovers, optimizes, and compresses convolutional neural network (CNN) architectures** for image classification.

Instead of manually designing neural architectures, SENN represents each CNN as a **genetic individual encoded by a deterministic Architecture DNA** and evolves architectures over multiple generations using a **mutation-driven evolutionary process** combined with rigorous selection and evaluation.

The evolutionary process in SENN is based on:

- Mutation-based architecture evolution  
- Selection of high-performing architectures  
- Multi-objective optimization (accuracy vs efficiency)  
- Strict architecture reproducibility via DNA  
- Structured pruning for model compression  

Manual architecture design is entirely avoided.  
Genetic crossover and weight inheritance are **intentionally excluded** to preserve architectural validity, deterministic reconstruction, and implementation robustness.

As evolution progresses, SENN discovers CNN architectures that achieve **strong accuracy while remaining parameter- and computation-efficient**, without human intervention.


---


## ✨ Key Features

- 🔬 Mutation-driven Neural Architecture Search (NAS) 
- 🧬 Deterministic Architecture DNA (JSON-based genotype) 
- ⚖️ Multi-objective optimization using Pareto dominance  
- 🔁 Structured pruning for efficiency improvement
- ✂️ Safe model construction with pre-training validation     
- 📊 Full lineage and mutation tracking   
- 📈 Reproducible architecture reconstruction 
- 🖥️ Optional Streamlit dashboard for live monitoring

---

## 🗂️ Directory Structure

```text
SENN/
├── data/               # CIFAR-10 data loaders and preprocessing
├── evaluation/         # PyTorch training and evaluation loops
├── evolution/          # Genetic algorithm, DNA schema, and mutation logic
├── frontend/           # React/Vite dashboard UI
├── models/             # PyTorch model definitions and baselines
├── pruning/            # Structural pruning utilities
├── scripts/            # Utility and experimental scripts
├── tests/              # Test cases and debugging scripts
├── utils/              # Latency, FLOPs calculation, and pareto logic
├── main.py             # Entry point for evolution process
├── server.py           # FastAPI backend
├── dashboard.py        # Streamlit dashboard
├── export_onnx.py      # ONNX export utility
└── config.py           # Global configuration
```

## 🚀 Quick Start

1. **Install dependencies:**  
   `pip install -r requirements.txt`
2. **Setup environment:**  
   Copy `.env.example` to `.env` and set your variables.
3. **Run Evolution Engine:**  
   `python main.py`
4. **Start API Server:**  
   `python server.py`
5. **Launch Dashboard:**  
   `streamlit run dashboard.py`

---

## 🧠 Core Idea

SENN evolves CNN architectures instead of hand-designing them.

High-level workflow:

```bash
Generate architectures
→ Train briefly
→ Evaluate
→ Select
→ Mutate DNA
→ Next generation
→ Repeat
```

Over generations, the population improves just like biological evolution — discovering architectures that balance **performance and efficiency**.

---

## 📂 Dataset & Preprocessing

### Dataset
- **Primary:** CIFAR-10  
  - 32×32 RGB images  
  - 10 classes  
- **Optional Extension:** CIFAR-100  

### Preprocessing Pipeline
- Tensor conversion  
- Normalization  
- Optional data augmentation:
  - Random crop
  - Horizontal flip  

### Data Splits
- Training set  
- Validation set  
- Test set  

---

## 🧬 Architecture Search Space (CNN DNA)

Each CNN is encoded as a **genotype (architecture DNA)** stored in JSON format.

### CNN Constraints

| Component | Options |
|--------|--------|
| Conv layers | 2 – 6 |
| Filters | 16 / 32 / 64 / 128 |
| Kernel sizes | 3×3, 5×5 |
| Activations | ReLU, LeakyReLU |
| Pooling | MaxPool, AvgPool, None |
| Normalization | Optional BatchNorm |
| Regularization | Optional Dropout |
| Head | Global Average Pooling + Dense |
| Model size | Small–medium CNNs |

The constrained search space ensures **valid, trainable architectures** while allowing rich diversity.

---

## ⚙️ Evolution Configuration

| Parameter | Typical Value |
|--------|--------|
| Population size | 8–12 |
| Generations | 10–20 |
| Survivors | Top-K / Pareto front |
| Training per model | 2–3 epochs |
| Total models evaluated | 100+ |

Short training during evolution allows efficient evaluation of many architectures.

---

## 🏋️ Training Strategy

### During Evolution
- Few epochs (2–3)
- Goal: estimate architectural potential
- Prevents overfitting and saves compute

### After Evolution
- Best architecture(s) fully trained
- 30–50 epochs
- Final evaluation on test set

---

## 📐 Fitness & Evaluation Metrics

SENN uses **multi-objective evaluation**.

### Primary Metrics
- Validation accuracy  
- Validation loss  

### Efficiency Metrics
- Number of parameters  
- FLOPs  
- Inference latency (optional)  

### Fitness Logic
- Early generations: weighted fitness  
- Later generations: Pareto optimization  

This prevents evolution from favoring large, inefficient models.

---

## 🏆 Selection Mechanisms

### Basic Selection
- Rank by fitness
- Select top-K models

### Advanced Selection
- Pareto front extraction
- **NSGA-II**
  - Non-dominated sorting
  - Crowding distance for diversity

Selected models become **parents** for the next generation.

---

## 🔁 Mutation Engine (Core Evolution)

Mutation introduces controlled randomness.

### Structural Mutations
- Add / remove convolution layers  
- Increase / decrease filters  
- Change kernel sizes  
- Toggle BatchNorm / Dropout  
- Change pooling strategy  
- Modify dense layer size  
- Adjust learning rate  

All mutations are **constraint-aware**, ensuring valid CNNs.

---

## ✂️ Pruning (Model Compression)

SENN integrates pruning for efficiency.

### Pruning Strategies
- Filter/channel reduction via mutation  
- L1-norm based channel pruning  
- Post-training pruning on survivors  

Result: **smaller, faster models with minimal accuracy loss**.

---

## 🔄 Full Evolution Loop

### Initialize population
→ Train

→ Measure (accuracy, params, FLOPs, latency)

→ Select (Pareto / NSGA-II)

→ Mutate DNA

→ Prune

→ Validate

→ Next generation


Repeated for **N generations**.

---

## 🏁 Final Model Selection

At the end of evolution:

- Extract Pareto-optimal architectures  
- Fully train best candidates  
- Evaluate on test set  

### Example Result
- CIFAR-10 accuracy: ~77–80%+  
- Reduced parameters and FLOPs vs baseline CNN  

---

## 📁 Outputs & Artifacts

### Model Files
- `best_model.pth`  
- `best_arch.json`  

### Logs
- `metrics.csv` — per-architecture metrics (val_accuracy, param_count, flops, latency)  
- `lineage.csv` — parent → child per generation  
- `mutation_history.json` — mutation log  
- `search_cost.json` — total architectures evaluated, wall time (for paper reporting)  

### Final evaluation (after training to convergence)
- `final_metrics.json` — test accuracy, param_count, flops (from `scripts/train_best_to_convergence.py`)  

### Visualizations
- Accuracy vs generation  
- Pareto fronts  
- Params/FLOPs vs accuracy  
- Confusion matrix  
- Training curves  

---

## 🖥️ Dashboard & Demo (Optional)

A **Streamlit dashboard** provides:

- Live evolution progress  
- Best architecture summary  
- Pareto front visualization  
- Architecture comparison table  
- Download links for models & DNA  

This transforms SENN from a research prototype into a **usable system**.

---

## 📐 Reproduction & Paper Evaluation

For **reproducible, paper-style results**:

### Determinism
- Set `seed` in `config.py` (or `SENN_SEED` env when running `main.py`). Same seed ⇒ same evolution and same best architecture.
- CuDNN determinism is enabled in `main.py` for reproducible training.

### Final training (convergence)
After evolution, train the best architecture to convergence with cosine LR:
```bash
python scripts/train_best_to_convergence.py --run_dir outputs/run_YYYYMMDD_HHMMSS [--epochs 200]
```
Writes `final_metrics.json` (test accuracy, params, FLOPs) and `best_model_trained_to_convergence.pth`.

### Multi-seed (mean ± std)
Run evolution + final training for several seeds and report mean ± std:
```bash
python scripts/run_multi_seed.py --num_seeds 5 --epochs_final 200 --out outputs/multi_seed_results.json
```
Uses `SENN_SEED` per seed when invoking `main.py`.

### Baselines
Train fixed architectures on the same data and recipe (cosine LR, same epochs):
```bash
python scripts/run_baselines.py --baselines small_vgg,small_resnet --epochs 200 --out outputs/baseline_results.json
```
Defined in `models/baselines.py` (SmallVGG, SmallResNet for CIFAR-10).

### Ablations (config)
In `config.py` (or by extending the CLI):
- `use_pruning: bool = True` — set `False` to disable pruning during search.
- `use_pareto: bool = True` — set `False` for fitness-only top-k selection (no Pareto/NSGA-II).

### One-command reproduction
- **Quick check:** `python scripts/reproduce_paper.py` (2 seeds, short final training).
- **Full table:** `python scripts/reproduce_paper.py --full` (5 seeds, 200 epochs final + baselines).

### Search cost
Each run writes `outputs/run_*/search_cost.json`: `total_architectures_evaluated`, `search_wall_time_seconds`. Use for reporting search efficiency in papers.

---

## 🛠️ Phase-Wise Implementation Plan

### Phase 0 – Baseline Evolution (MVP)
- Population
- Mutation
- Selection
- Training loop

### Phase 1 – Architecture DNA
- JSON genotype
- Safe model builder
- Logging

### Phase 2 – Multi-Objective Optimization
- Pareto fronts
- NSGA-II

### Phase 3 – Efficiency Metrics
- Params
- FLOPs
- Latency

### Phase 4 – Pruning
- Structured compression

### Phase 5 – Dashboard
- Visualization & interaction





