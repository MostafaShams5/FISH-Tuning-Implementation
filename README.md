# FISH-Tuning: An Implementation

This repository provides a hands-on implementation of the methods described in the research paper **"FISH-Tuning: Enhancing PEFT Methods with Fisher Information"**.  
My goal with this project was to faithfully reproduce the paper's core methodology, validate its claims through direct experimentation, and provide a clear framework for others to use and build upon.

---

## Idea Behind FISH-Tuning (short & technical)

Training large language models is computationally intensive. Parameter-Efficient Fine-Tuning (PEFT) methods like **LoRA** (Low-Rank Adaptation) reduce cost by freezing most pre-trained parameters and adding a small set of trainable parameters (e.g., low-rank matrices).  
**FISH-Tuning** observes: those *added* parameters are themselves candidates for further sparsification. By estimating each added parameter's importance using **Fisher Information**, FISH-Tuning constructs a binary mask to update only the most important added parameters during training — reducing updates and possibly improving generalization while keeping compute nearly the same.

**Quick math intuition (brief):** Fisher Information measures how much a parameter affects the model's output distribution (via gradients of the log-likelihood). Parameters with large Fisher score contribute more to predictive uncertainty and are therefore better candidates for selective updating.

---

## Repository Structure

```
FISH-Tuning-Project/
├── configs/
│   ├── base_config.yaml
│   └── experiments/
│       ├── bert_sst2.yaml
│       └── smoke_test.yaml
├── scripts/
│   └── run_experiment.py
└── src/
    └── fish_tuning/
        ├── __init__.py
        ├── fisher_utils.py
        └── trainer.py
```

---

## Experiments: Reproducing Paper Claims (summary)

Primary test: fine-tuning `bert-base-cased` on **SST-2** to compare with the paper's reported results.

### Claim 1 — Superior performance with same parameter count
My experiment (parameter count matched at **443,906**) — SST-2 accuracy:

| Method            | My Accuracy | Paper's Accuracy (Table 6) | Verdict    |
|-------------------|-------------:|---------------------------:|------------|
| Original LoRA     | 90.48%      | 90.14%                     |            |
| LoRA-FISH (Mine)  | 90.94%      | 90.71%                     | ✅ Confirmed |

**Interpretation:** LoRA-FISH yields a modest but consistent accuracy increase versus standard LoRA when the number of trainable parameters is held equal.

**Visualization:** See `assets/accuracy_comparison.png`.

---

### Claim 2 — The importance of Fisher Information (contrastive study)

| Method             | Selection Method       | Accuracy |
|--------------------|------------------------|---------:|
| LoRA-FISH (Mine)   | Highest Fisher Score   | 90.94%  |
| LoRA-FISH-rand     | Random                 | 90.25%  |
| LoRA-FISH-rev      | Lowest Fisher Score    | 74.43%  | ✅ Confirmed

**Visualization:** See `assets/contrastive_study.png`.

---

### Claim 3 — No significant increase in training cost

My resource numbers (Kaggle T4 GPU):

| Method            | Train Time (s) | Peak GPU Memory (MB) | Verdict |
|-------------------|---------------:|---------------------:|--------:|
| Original LoRA     | 2303.90        | 1575.23              |         |
| LoRA-FISH (Mine)  | 2328.78        | 1604.19              | ✅ Confirmed (negligible diff) |

**Visualization:** See `assets/training_time.png` and `assets/gpu_memory.png`.

---

## How to run experiments

1. Clone the repo:
```bash
git clone https://github.com/your-username/FISH-Tuning-Project.git
cd FISH-Tuning-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure an experiment YAML under `configs/experiments/` (e.g., `bert_sst2.yaml`).

4. Run the experiment:
```bash
python scripts/run_experiment.py --config configs/experiments/bert_sst2.yaml
```

---

## Visualizations (included)

I added a small set of publication-ready charts showing:

- SST-2 accuracy comparisons across methods (`assets/accuracy_comparison.png`)
- Contrastive study (selection methods) (`assets/contrastive_study.png`)
- Training time comparison (`assets/training_time.png`)
- Peak GPU memory comparison (`assets/gpu_memory.png`)

---

## Links & Notebooks

- Paper (arXiv): https://arxiv.org/abs/2504.04050  
- My Kaggle experiment (BERT on SST-2): https://www.kaggle.com/code/shamsccs/bertsst2/notebook  
- My Kaggle smoke test: https://www.kaggle.com/code/shamsccs/prajjwal-tinymodel-smoketest

---

## Citation

If you use this code or its concepts in your research, please cite the original paper:

```bibtex
@misc{xue2025fishtuning,
  title={FISH-Tuning: Enhancing PEFT Methods with Fisher Information},
  author={Kang Xue, Ming Dong, Xinhui Tu, Tingting He},
  year={2025},
  eprint={2504.04050},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---

## License

Add your preferred license here (e.g., MIT).
