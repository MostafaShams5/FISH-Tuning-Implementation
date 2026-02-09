# FISH-Tuning: An Implementation of "Enhancing PEFT Methods with Fisher Information"

This repository contains a PyTorch implementation of **FISH-Tuning**, a powerful technique from the paper [FISH-Tuning: Enhancing PEFT Methods with Fisher Information](https://arxiv.org/abs/2504.04050).

The core idea is based on making already-efficient tuning methods even smarter. Many popular Parameter-Efficient Fine-Tuning (PEFT) methods work by adding a small number of new, trainable parameters to a large frozen model. While this saves a lot of memory, these methods usually train *all* of the new parameters.

This means that even within these small, efficient modules, some parameters are more important than others for learning a new task. FISH-Tuning uses the **Fisher Information Matrix (FIM)** to figure out which parameters are the most critical. Think of Fisher Information as a way to measure a parameter's influence—a high score means changing that parameter will have a big effect on the model's predictions.

A visual comparison of standard LoRA versus LoRA-FISH is shown below. While standard LoRA trains dense adapter matrices, FISH-Tuning identifies and freezes less important parameters (grey boxes), resulting in sparse updates focused only on critical weights.

<p align="center">
  <img src="Original LoRA vs LoRA-FISH.png" alt="Comparison of Original LoRA vs LoRA-FISH" width="800">
  <br>
  <em>Figure 1: Visual comparison from the paper showing Original LoRA (left) where all adapter parameters are trainable, and LoRA-FISH (right) where only parameters with high Fisher information are trained.</em>
</p>

FISH-Tuning calculates these scores for all the new PEFT parameters and then creates a "mask" to freeze the least important ones. This allows the model to focus its training effort only on the parameters that matter most, leading to a more effective and targeted fine-tuning process.

This allows us to:

1. Achieve **better performance** than standard PEFT methods while using the same number of trainable parameters.
2. Drastically **reduce the number of trainable parameters** with minimal impact on performance, saving resources.

This implementation is built on top of Hugging Face `transformers` and `peft`, making it easy to use in your own projects.

## Experiments and Results

We've run several experiments to show that this method works just as well in practice as it does in theory. The results consistently prove that selecting parameters based on Fisher information is a winning strategy.

### 1. Replicating the Paper's Findings (BERT on SST2)

Our first goal was to confirm the results from the original paper. We ran an experiment fine-tuning a `bert-base-cased` model on the `sst2` dataset. We compared four approaches:

* **Original LoRA**: The standard baseline.
* **LoRA-FISH (Ours)**: Our implementation, using Fisher scores to guide training.
* **LoRA-FISH-rand**: A control group where we randomly pick which parameters to train.
* **LoRA-FISH-rev**: A control group where we train the *least* important parameters.

To make it a fair fight, all masked methods (`-FISH`, `-rand`, `-rev`) were set up to have the **exact same number of trainable parameters** as the Original LoRA baseline.

**Our Results:**

| Method               | Train Time (s) | Peak GPU Mem   | Final Trainable Params | Val Accuracy |
| :------------------- | :------------- | :------------- | :--------------------- | :----------- |
| Original LoRA        | 2303.90        | 1575.23 MB     | 443,906                | 0.9048       |
| **LoRA-FISH (Ours)** | **2328.78** | **1604.19 MB** | **443,906** | **0.9094** |
| LoRA-FISH-rand       | 2328.83        | 1603.69 MB     | 443,906                | 0.9025       |
| LoRA-FISH-rev        | 2323.71        | 1603.69 MB     | 443,906                | 0.7443       |

You can clearly see that **LoRA-FISH (Ours) achieved a higher accuracy than original LoRA**. The fact that the random method did worse and the reverse-importance method failed badly proves that the Fisher score is successfully identifying the parameters that truly matter.

#### **Comparison with the Original Paper**

Our results show the exact same pattern as the paper's own contrastive study (Table 3). Here is the data from their experiments on the GLUE benchmark:

| Method (from paper) | Trainable Params | GLUE Avg Score |
| :------------------ | :--------------- | :------------- |
| Original-LoRA       | 0.0057%          | 68.45          |
| **LoRA-FISH** | **0.0057%** | **68.90** |
| LoRA-FISH-rand      | 0.0057%          | 68.74          |
| LoRA-FISH-rev       | 0.0057%          | 66.71          |

In both our experiment and the paper's, the conclusion is identical: **FISH > rand > rev**. Selecting parameters by Fisher score is the best approach, while selecting the *least* important parameters is the worst. This confirms our implementation is behaving exactly as expected.

> **Reproduction Proof:** You can view the full execution logs, code, and final metrics of this exact experiment in our [runnable Kaggle Notebook](https://www.kaggle.com/code/shamsccs/bertsst2).

#### A Note on Resources and Replication

Please keep in mind that due to the resource limitations of free notebooks on Kaggle, we couldn't replicate every experiment from the paper, such as running on the full GLUE benchmark. Instead, we focused on key, representative tasks to validate the core ideas.

You may also notice in our results that LoRA-FISH can use slightly more peak GPU memory than the baseline LoRA. **This is expected behavior.** As the original paper explains in section 5.4.4, this small increase is because the model needs to store the generated gradient mask in memory during training.

### 2. Pushing the Boundaries with Fewer Parameters

Next, we tested the main promise of FISH-Tuning: can we train with far fewer parameters and still get good results?

#### TinyLlama on Rotten Tomatoes (25% of Parameters)

We fine-tuned `TinyLlama-1.1B` but configured LoRA-FISH to use only **25% of the trainable parameters** of the baseline LoRA.

**Results:**

| Method               | Train Time (s) | Peak GPU Mem    | Final Trainable Params | Val Accuracy |
| :------------------- | :------------- | :-------------- | :--------------------- | :----------- |
| Original LoRA        | 4938.97        | 12021.40 MB     | 2,256,896              | 0.9006       |
| **LoRA-FISH (Ours)** | **4959.14** | **12233.47 MB** | **564,224** | **0.8856** |

With a **4x reduction in trainable parameters**, the accuracy only dropped by a minor 1.5%. This demonstrates the efficiency of FISH-Tuning.

> You can find the full experiment notebook on Kaggle: [TinyLlama-FISH-Tuning (25% ratio)](https://www.kaggle.com/code/shamsccs/tinyllama-fish/).

#### BERT-tiny on SST2 (50% of Parameters)

We ran a similar test on a much smaller model (`bert-tiny`) and used **50% of the baseline parameters**.

**Results:**

| Method               | Train Time (s) | Peak GPU Mem | Final Trainable Params | Val Accuracy |
| :------------------- | :------------- | :----------- | :--------------------- | :----------- |
| Original LoRA        | 4.20           | 68.53 MB     | 6,402                  | 0.5103       |
| **LoRA-FISH (Ours)** | **1.61** | **76.94 MB** | **3,201** | **0.5069** |

Once again, the results were favorable. We **halved the trainable parameters** with almost no change in performance.

> You can find the full experiment notebook on Kaggle: [BERT-tiny Smoke Test](https://www.kaggle.com/code/shamsccs/prajjwal-tinymodel-smoketest).

### Technical Insights & Trade-offs

While FISH-Tuning offers superior parameter efficiency, it introduces specific engineering considerations that we observed during implementation:

1.  **Pre-computation Overhead:** Before training begins, there is a one-time computational cost to pass a calibration dataset through the model to calculate Fisher scores. On Kaggle T4 GPUs, this took approximately 2-3 minutes for BERT-base.
2.  **Peak Memory Usage:** As noted in the paper and confirmed in our logs, peak GPU memory during training is slightly higher than standard LoRA. This is because the binary mask for the gradients must be stored in GPU memory alongside the model weights. The increase is minor but relevant for constrained environments.

## How to Use This Repository

The project is built to be easy to use. Here's how you can get started.

### Step 1: Clone the Repository

First, get the code on your local machine.

```bash
git clone https://github.com/MostafaShams5/FISH-Tuning-Implementation.git
cd FISH-Tuning-Implementation
```

### Step 2: Set Up Your Environment

We recommend using a virtual environment to keep your packages organized.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (on Linux/macOS)
source venv/bin/activate

# Install the project in editable mode along with its dependencies
pip install -e .
```

Make sure you have PyTorch installed with CUDA support if you want to use a GPU.

### Step 3: Run an Experiment

You have two main ways to use this code.

**Path A: The Quick Way (Using Config Files)**

This is the fastest way to run a full experiment comparing LoRA and LoRA-FISH. Just create a YAML file in `configs/experiments/` and let our script handle the rest.

For example, to run your own test, create `configs/experiments/my_experiment.yaml`:

```yaml
model:
  name: "roberta-base"
dataset:
  load_args: ["glue", "mrpc"]
lora:
  baseline_rank: 8
  fish_rank: 16 # Use a larger rank to give more parameters to choose from
fish_tuning:
  prune_to_ratio_of_baseline: 0.5 # We want the final model to have 50% of baseline's params
```

Then, launch the experiment from your terminal:

```bash
python scripts/run_experiment.py --config configs/experiments/my_experiment.yaml
```

**Path B: The Engineer's Way (Integrating into Your Project)**

For full control, you can integrate FISH-Tuning directly into your own training code. The `src/fish_tuning` directory contains everything you need. Here is a guide on how to use the core components in your own script.

```python
# In your own training script (e.g., my_training_script.py)

# === Step 1: Import the FISH-Tuning utilities ===
# Assuming the 'src' directory is in your Python path, you can import these.
# You can also just copy the src/fish_tuning folder into your project.
from fish_tuning import (
    calculate_fisher_information,
    create_mask_from_fisher,
    FishTuningTrainer,
)

# === Step 2: Prepare your model and a calibration dataloader ===
# This part is standard setup. You need a PEFT-enabled model
# and a small dataloader with a sample of your training data.
# It's best to use a larger LoRA rank here (e.g., 32) to create a
# rich pool of parameters for the algorithm to choose from.
#
# peft_model = get_peft_model(...)
# calibration_dataloader = DataLoader(...)

# === Step 3: Calculate Fisher Scores ===
# This function loops through the calibration data to find the
# importance score for each trainable parameter.
print("Calculating Fisher Information...")
fisher_scores = calculate_fisher_information(
    model=peft_model,
    calibration_dataloader=calibration_dataloader,
    device=device,
)

# === Step 4: Create the Pruning Mask ===
# This utility takes the scores and creates a binary mask.
# You specify what fraction of parameters you want to keep training.
print("Creating sparsity mask...")
mask = create_mask_from_fisher(
    fisher_scores=fisher_scores,
    keep_ratio=0.5,  # Keep the top 50% of parameters
    names_to_exclude=['classifier'], # Optional: protect layers from being pruned
)

# === Step 5: Use the Custom Trainer ===
# The FishTuningTrainer works just like the standard Hugging Face Trainer.
# You just need to pass the mask you created.
print("Initializing FishTuningTrainer...")
trainer = FishTuningTrainer(
    model=peft_model,
    args=training_args,
    mask=mask,  # <-- This is the key part!
    # ...add your other standard arguments like train_dataset, etc.
)

# Now you can train your model as usual. The trainer will handle
# applying the mask to the gradients automatically.
trainer.train()
```

This approach allows you to seamlessly add the intelligence of FISH-Tuning to any existing fine-tuning workflow.

## Citation

If you use this work in your research, please cite the original paper:

```bibtex
@misc{xue2025fishtuning,
      title={FISH-Tuning: Enhancing PEFT Methods with Fisher Information}, 
      author={Kang Xue and Ming Dong and Xinhui Tu and Tingting He},
      year={2025},
      eprint={2504.04050},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
