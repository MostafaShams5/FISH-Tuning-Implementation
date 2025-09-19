# FISH-Tuning: An Implementation

This repository provides a hands-on implementation of the methods described in the research paper [**FISH-Tuning: Enhancing PEFT Methods with Fisher Information**](https://arxiv.org/abs/2504.04050).

My goal with this project was to faithfully reproduce the paper's core methodology, validate its claims through direct experimentation, and provide a clear framework for others to use and build upon.

## The Idea Behind FISH-Tuning

Training large language models is computationally intensive. Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA (Low-Rank Adaptation) were created to solve this. LoRA works by freezing the vast majority of a pre-trained model's parameters and injecting a small number of new, trainable parameters, typically in the form of low-rank matrices.

However, the authors of FISH-Tuning identified a further opportunity for optimization.

**Paper's Core Insight (from the Abstract):** "While addition-based and reparameterization-based PEFT methods like LoRA and Adapter already fine-tune only a small number of parameters, the newly introduced parameters within these methods themselves present an opportunity for further optimization. Selectively fine-tuning only the most impactful among these new parameters could further reduce resource consumption while maintaining, or even improving, fine-tuning effectiveness."

To identify these "most impactful" parameters, the paper proposes using Fisher Information as a proxy for parameter importance. By calculating a score for each new parameter introduced by LoRA, a binary mask can be created. This mask is then used during training to ensure that only the highest-scoring, most critical parameters receive gradient updates.

This project implements that exact process, which I call **LoRA-FISH**.

---

## Repository Structure

```
FISH-Tuning-Project/
├── configs/
│   ├── base_config.yaml          # Default training hyperparameters
│   └── experiments/
│       ├── bert_sst2.yaml        # Main experiment config file
│       └── smoke_test.yaml       # A very fast test config for validation
├── scripts/
│   └── run_experiment.py         # The main script to launch training and evaluation
└── src/
    └── fish_tuning/
        ├── __init__.py           # Makes the code importable as a package
        ├── fisher_utils.py       # Core logic for calculating Fisher scores and creating masks
        └── trainer.py            # The custom Hugging Face Trainer that applies the gradient mask
```

---

## Experiments: Validating the Paper's Claims

I conducted experiments to verify the key claims made in the paper. My primary test case was fine-tuning the `bert-base-cased` model on the SST-2 sentiment classification task, which allows for a direct comparison with the results published in the paper's appendix.

You can view and run my full experiments in these Kaggle notebooks:

- **BERT SST-2 experiment (full)**: https://www.kaggle.com/code/shamsccs/bertsst2/notebook
- **Prajjwal tiny-model smoke test**: https://www.kaggle.com/code/shamsccs/prajjwal-tinymodel-smoketest

---

## Claim 1: Superior Performance with the Same Parameter Count

**Paper's Claim (from the Abstract):** "Experimental results... demonstrate that FISH-Tuning consistently outperforms the vanilla PEFT methods when using the same proportion of trainable parameters."

I configured my experiment so that both the standard Original LoRA and my LoRA-FISH model had the exact same number of trainable parameters (443,906).

**My Results vs. Paper's Results (SST-2 Accuracy):**

| Method | My Accuracy | Paper's Accuracy (Table 6) | Verdict |
|---|---:|---:|:--:|
| Original LoRA | 90.48% | 90.14% | |
| LoRA-FISH (Mine) | 90.94% | 90.71% | ✅ Confirmed |

**Analysis:** My results strongly validate this claim. LoRA-FISH achieved a higher accuracy than the standard LoRA baseline, and my scores are numerically consistent with those reported in the paper, confirming a tangible performance improvement.

---

## Claim 2: The Importance of Fisher Information

**Paper's Rationale (from Section 5.3.3, "Contrastive Study"):** To prove that the Fisher score is the active ingredient, the paper compares its method against selecting parameters randomly or in reverse order of importance.

I replicated this "contrastive study" to test the significance of the selection criteria.

**My Results:**

| Method | Selection Method | Accuracy | Verdict |
|---|---|---:|:--:|
| LoRA-FISH (Mine) | Highest Fisher Score | 90.94% | |
| LoRA-FISH-rand | Random | 90.25% | |
| LoRA-FISH-rev | Lowest Fisher Score | 74.43% | ✅ Confirmed |

**Analysis:** The catastrophic drop in performance for LoRA-FISH-rev is the most telling result. It demonstrates that the Fisher score provides a meaningful and accurate measure of parameter importance. Training the lowest-scoring parameters is significantly worse than doing nothing, proving the method's validity.

---

## Claim 3: No Significant Increase in Training Cost

**Paper's Claim (from the Abstract):** "FISH-Tuning aims to achieve superior performance without increasing training time or inference latency..."

**My Results:**

| Method | My Train Time | My Peak GPU Memory | Verdict |
|---|---:|---:|:--:|
| Original LoRA | 2303.90 s | 1575.23 MB | |
| LoRA-FISH (Mine) | 2328.78 s | 1604.19 MB | ✅ Confirmed |

**Analysis:** The training time difference is negligible (~1%), confirming that applying the mask adds no significant computational overhead. The slight increase in GPU memory is also consistent with the paper's explanation that memory is needed to store the mask tensor itself.

---

## A Note on My Experiments

All tests for this project were run using the free T4 GPU provided by Kaggle. Due to these hardware limitations, I focused on meticulously reproducing one of the paper's key experiments (`bert-base-cased` on SST-2) rather than all of them. My results confirm that this implementation successfully replicates the paper's findings on this benchmark.

---

## How to Use This Project

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/FISH-Tuning-Project.git
cd FISH-Tuning-Project

# 2. Install the necessary dependencies
pip install -r requirements.txt
```

### Running an Experiment

All experiments are controlled via YAML files in the `configs/` directory.

**Configure:** Modify an existing `.yaml` file in `configs/experiments/` or create a new one to define your model, dataset, and training parameters.

**Execute:** Run the main script from the terminal, pointing it to your configuration file.

```bash
# Run the main BERT on SST-2 experiment
python scripts/run_experiment.py --config configs/experiments/bert_sst2.yaml
```

---

## Kaggle Notebooks

- BERT SST-2 (full experiment): https://www.kaggle.com/code/shamsccs/bertsst2/notebook
- Prajjwal tiny-model smoke test: https://www.kaggle.com/code/shamsccs/prajjwal-tinymodel-smoketest

---

## Citation

This project is an implementation of the work by Kang Xue, Ming Dong, Xinhui Tu, and Tingting He. If you use this code or its concepts in your research, please cite the original paper:

```bibtex
@misc{xue2025fishtuning,
      title={FISH-Tuning: Enhancing PEFT Methods with Fisher Information},
      author={Kang Xue, Ming Dong, Xinhui Tu, and Tingting He},
      year={2025},
      eprint={2504.04050},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.04050}
}
```

---

If you'd like, I can also update the repository's `README.md` on GitHub (if you give me the repo link and permission), or convert this document into a ready-to-commit `README.md` file. Let me know which you prefer.
