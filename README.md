# FISH-Tuning: An Implementation

This repository provides a hands-on implementation of the methods described in the research paper **"[FISH-Tuning: Enhancing PEFT Methods with Fisher Information](https://arxiv.org/abs/2504.04050)"**.

My goal with this project was to faithfully reproduce the paper's core methodology, validate its claims through direct experimentation, and provide a clear framework for others to use and build upon.

## The Idea Behind FISH-Tuning

When we fine-tune an AI model, methods like **LoRA** help save time and computer power by only training a few small, new parts of the model.

The idea of FISH-Tuning is to take this one step further. It finds the most important, or "impactful," parameters within those small new parts and focuses all the training effort only on them. It uses a mathematical tool called **Fisher Information** to score each parameter's importance. By training only the high-scoring parameters, the model can learn more efficiently and effectively.

## Repository Structure

The project is organized to separate the core logic from the experimental scripts for clarity and reusability.

FISH-Tuning-Project/
├── configs/
│ ├── base_config.yaml # Default training hyperparameters
│ └── experiments/
│ ├── bert_sst2.yaml # Main experiment config file
│ └── smoke_test.yaml # A very fast test config for validation
├── scripts/
│ └── run_experiment.py # The main script to launch training and evaluation
└── src/
└── fish_tuning/
├── init.py # Makes the code importable as a package
├── fisher_utils.py # Core logic for calculating Fisher scores and creating masks
└── trainer.py # The custom Hugging Face Trainer that applies the gradient mask

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
## Experiments: Validating the Paper's Claims

I conducted experiments to verify the key claims made in the paper. My primary test case was fine-tuning the `bert-base-cased` model on the SST-2 sentiment classification task, which allows for a direct comparison with the results published in the paper's appendix.

**You can view and run my full experiment in this [Kaggle Notebook](https://www.kaggle.com/code/shamsccs/bertsst2/notebook).**

### Claim 1: Superior Performance with the Same Parameter Count

> **Paper's Claim (from the Abstract):** *"Experimental results... demonstrate that FISH-Tuning **consistently outperforms the vanilla PEFT methods** when using the same proportion of trainable parameters."*

I configured my experiment so that both the standard `Original LoRA` and my `LoRA-FISH` model had the exact same number of trainable parameters (443,906).

**My Results vs. Paper's Results (SST-2 Accuracy):**

| Method             | My Accuracy | Paper's Accuracy (Table 6) | Verdict |
| ------------------ | ----------- | -------------------------- | ------- |
| Original LoRA      | 90.48%      | 90.14%                     |         |
| **LoRA-FISH (Mine)**| **90.94%**  | **90.71%**                 | ✅ **Confirmed** |

**Analysis:** My results strongly validate this claim. LoRA-FISH achieved a higher accuracy than the standard LoRA baseline, and my scores are numerically consistent with those reported in the paper, confirming a tangible performance improvement.

### Claim 2: The Importance of Fisher Information

> **Paper's Rationale (from Section 5.3.3, "Contrastive Study"):** To prove that the Fisher score is the active ingredient, the paper compares its method against selecting parameters randomly or in reverse order of importance.

I replicated this "contrastive study" to test the significance of the selection criteria.

**My Results:**

| Method             | Selection Method         | Accuracy | Verdict |
| ------------------ | ------------------------ | -------- | ------- |
| **LoRA-FISH (Mine)**| **Highest** Fisher Score | **90.94%** |         |
| LoRA-FISH-rand     | Random                   | 90.25%   |         |
| LoRA-FISH-rev      | **Lowest** Fisher Score  | 74.43%   | ✅ **Confirmed** |

**Analysis:** The catastrophic drop in performance for `LoRA-FISH-rev` is the most telling result. It demonstrates that the Fisher score provides a meaningful and accurate measure of parameter importance. Training the lowest-scoring parameters is significantly worse than doing nothing, proving the method's validity.

### Claim 3: No Significant Increase in Training Cost

> **Paper's Claim (from the Abstract):** *"FISH-Tuning aims to achieve superior performance **without increasing training time or inference latency**..."*

**My Results:**

| Method             | My Train Time | My Peak GPU Memory | Verdict |
| ------------------ | ------------- | ------------------ | ------- |
| Original LoRA      | 2303.90 s     | 1575.23 MB         |         |
| **LoRA-FISH (Mine)**| **2328.78 s** | **1604.19 MB**     | ✅ **Confirmed** |

**Analysis:** The training time difference is negligible (~1%), confirming that applying the mask adds no significant computational overhead. The slight increase in GPU memory is also consistent with the paper's explanation that memory is needed to store the mask tensor itself.

### A Note on My Experiments

All tests for this project were run using the free T4 GPU provided by Kaggle. Due to these hardware limitations, I focused on meticulously reproducing one of the paper's key experiments (`bert-base-cased` on SST-2) rather than all of them. My results confirm that this implementation successfully replicates the paper's findings on this benchmark.

## How to Use This Project

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/FISH-Tuning-Project.git
cd FISH-Tuning-Project

# 2. Install the necessary dependencies
pip install -r requirements.txt
Running an Experiment

All experiments are controlled via YAML files in the configs/ directory.

Configure: Modify an existing .yaml file in configs/experiments/ or create a new one to define your model, dataset, and training parameters.

Execute: Run the main script from the terminal, pointing it to your configuration file.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# Run the main BERT on SST-2 experiment
python scripts/run_experiment.py --config configs/experiments/bert_sst2.yaml
Citation

This project is an implementation of the work by Kang Xue, Ming Dong, Xinhui Tu, and Tingting He. If you use this code or its concepts in your research, please cite the original paper.

code
Bibtex
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
@misc{xue2025fishtuning,
      title={FISH-Tuning: Enhancing PEFT Methods with Fisher Information},
      author={Kang Xue, Ming Dong, Xinhui Tu, Tingting He},
      year={2025},
      eprint={2504.04050},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
License

This project is licensed under the MIT License. See the LICENSE file for details.
