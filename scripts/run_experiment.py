# scripts/run_experiment.py

import torch
import numpy as np
import pandas as pd
import yaml
import os
import tempfile
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from functools import partial

import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory by going one level up (e.g., /.../)
project_root = os.path.dirname(script_dir)
# Construct the path to the src directory
src_path = os.path.join(project_root, 'src')
# Add the src path to the system's module search path
sys.path.insert(0, src_path)
# --- END: ROBUST IMPORT FIX ---
from fish_tuning import (
    FishTuningTrainer,
    calculate_fisher_information,
    create_mask_from_fisher,
)

# =================================================================================
# UTILITY FUNCTIONS
# =================================================================================
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_keep_ratio(baseline_params, target_params):
    return baseline_params / target_params

# =================================================================================
# MAIN GENERIC EXPERIMENT RUNNER
# =================================================================================
def run_generic_experiment(config):
    """
    Main function to run an experiment based on a configuration dictionary.
    """
    cfg_model = config['model']
    cfg_dataset = config['dataset']
    cfg_lora = config['lora']
    cfg_training = config['training']
    cfg_fish = config['fish_tuning']

    mask_dir = tempfile.mkdtemp()
    print(f"Temporary directory for masks created at: {mask_dir}")

    # --- Load Dataset and Tokenizer ---
    print(f"Loading dataset '{cfg_dataset['name']}' and tokenizer for '{cfg_model['name']}'...")
    dataset = load_dataset(*cfg_dataset['load_args'])
    tokenizer = AutoTokenizer.from_pretrained(cfg_model['name'])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        cfg_model.setdefault('config_overrides', {})['pad_token_id'] = tokenizer.eos_token_id

    def preprocess_function(examples):
        return tokenizer(examples[cfg_dataset['text_column']], truncation=True, padding='max_length', max_length=cfg_training['max_seq_length'])

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    results = {}
    
    label_column = cfg_dataset.get('label_column', 'label')
    num_labels = len(dataset['train'].unique(label_column))
    
    # --- FIXED: DEFINE compute_metrics HERE ---
    # This function now correctly closes over the `config` variable.
    metric = load_metric('glue', cfg_dataset['name'])
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # =============================================================================
    # EXPERIMENT A: BASELINE (ORIGINAL LORA)
    # =============================================================================
    print("\n" + "="*50)
    print("RUNNING EXPERIMENT A: BASELINE (ORIGINAL LORA)")
    print("="*50)
    
    model_baseline = AutoModelForSequenceClassification.from_pretrained(
        cfg_model['name'], num_labels=num_labels, **cfg_model.get('config_overrides', {})
    )
    
    lora_config_baseline = LoraConfig(
        r=cfg_lora['baseline_rank'], lora_alpha=2 * cfg_lora['baseline_rank'],
        target_modules=cfg_lora['target_modules'], lora_dropout=cfg_lora['dropout'],
        bias="none", task_type=TaskType.SEQ_CLS,
    )
    
    model_baseline_peft = get_peft_model(model_baseline, lora_config_baseline)
    baseline_trainable_params = count_trainable_parameters(model_baseline_peft)
    print(f"Baseline LoRA (r={cfg_lora['baseline_rank']}) Trainable Parameters: {baseline_trainable_params:,}")
    
    # Use partial to merge base training args with experiment-specific ones
    training_args_dict = {**config['base_training']['args'], **cfg_training.get('args_override', {})}
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg_training['output_dir'], "baseline_lora"),
        **training_args_dict
    )
    
    trainer_baseline = Trainer(
        model=model_baseline_peft, args=training_args, train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[cfg_dataset['validation_split']], tokenizer=tokenizer,
        data_collator=data_collator, compute_metrics=compute_metrics,
    )
    
    trainer_baseline.train()
    eval_key = f"eval_{training_args.metric_for_best_model}"
    eval_results_baseline = trainer_baseline.evaluate()
    
    results['Original LoRA'] = {
        'Initial Rank': cfg_lora['baseline_rank'], 'Final Trainable Params': baseline_trainable_params,
        'Keep Ratio': '100%', f'Validation {training_args.metric_for_best_model.capitalize()}': eval_results_baseline[eval_key]
    }

    # =============================================================================
    # SETUP & RUN FOR MASKED METHODS
    # =============================================================================
    if cfg_fish['enabled']:
        print("\n" + "="*50)
        print("PRE-COMPUTATION FOR FISH, RAND, and REV METHODS")
        print("="*50)
        
        model_for_fish = AutoModelForSequenceClassification.from_pretrained(
            cfg_model['name'], num_labels=num_labels, **cfg_model.get('config_overrides', {})
        )
        
        lora_config_fish = LoraConfig(
            r=cfg_lora['fish_rank'], lora_alpha=2 * cfg_lora['fish_rank'],
            target_modules=cfg_lora['target_modules'], lora_dropout=cfg_lora['dropout'],
            bias="none", task_type=TaskType.SEQ_CLS,
        )
        
        model_for_fish_peft = get_peft_model(model_for_fish, lora_config_fish)
        fish_trainable_params = count_trainable_parameters(model_for_fish_peft)
        
        keep_ratio = calculate_keep_ratio(baseline_trainable_params, fish_trainable_params)
        
        calibration_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(cfg_fish['num_samples']))
        calibration_dataloader = DataLoader(calibration_dataset, batch_size=1, collate_fn=data_collator)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fisher_scores = calculate_fisher_information(
            model=model_for_fish_peft, calibration_dataloader=calibration_dataloader, device=device
        )

        names_to_exclude = cfg_lora.get('names_to_exclude', [])

        # --- FIXED: Fairer Mask Creation ---
        # 1. FISH Mask (based on actual Fisher scores)
        fish_mask = create_mask_from_fisher(fisher_scores, keep_ratio, names_to_exclude=names_to_exclude)
        
        # 2. Random Mask (based on random scores for all trainable params)
        random_scores = {name: torch.rand_like(p) for name, p in model_for_fish_peft.named_parameters() if p.requires_grad}
        random_mask = create_mask_from_fisher(random_scores, keep_ratio, names_to_exclude=names_to_exclude)
        
        # 3. Reverse Mask (based on negated Fisher scores)
        reverse_mask = create_mask_from_fisher({name: -s for name, s in fisher_scores.items()}, keep_ratio, names_to_exclude=names_to_exclude)

        masks_to_run = {
            "LoRA-FISH (Ours)": fish_mask,
            "LoRA-FISH-rand": random_mask,
            "LoRA-FISH-rev": reverse_mask,
        }

        for exp_name, mask in masks_to_run.items():
            print("\n" + "="*50)
            print(f"RUNNING EXPERIMENT: {exp_name}")
            print("="*50)
            
            model_to_train = AutoModelForSequenceClassification.from_pretrained(
                cfg_model['name'], num_labels=num_labels, **cfg_model.get('config_overrides', {})
            )
            model_to_train_peft = get_peft_model(model_to_train, lora_config_fish)
            
            # Re-use the training_args from the baseline run for consistency
            masked_training_args = TrainingArguments(
                output_dir=os.path.join(cfg_training['output_dir'], exp_name.lower().replace(" ", "_")),
                **training_args_dict
            )
            
            trainer_masked = FishTuningTrainer(
                model=model_to_train_peft, args=masked_training_args, mask=mask,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset[cfg_dataset['validation_split']],
                tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics,
            )

            trainer_masked.train()
            eval_key_masked = f"eval_{masked_training_args.metric_for_best_model}"
            eval_results_masked = trainer_masked.evaluate()
            
            results[exp_name] = {
                'Initial Rank': cfg_lora['fish_rank'], 'Final Trainable Params': baseline_trainable_params,
                'Keep Ratio': f"{keep_ratio:.2%}",
                f'Validation {masked_training_args.metric_for_best_model.capitalize()}': eval_results_masked[eval_key_masked]
            }
            
    # =============================================================================
    # PRINT FINAL RESULTS
    # =============================================================================
    print("\n" + "="*60)
    print(f"          FINAL EXPERIMENT RESULTS for {cfg_model['name']} on {cfg_dataset['name']}")
    print("="*60)
    
    df = pd.DataFrame.from_dict(results, orient='index')
    desired_order = ['Original LoRA', 'LoRA-FISH (Ours)', 'LoRA-FISH-rand', 'LoRA-FISH-rev']
    df = df.reindex([idx for idx in desired_order if idx in df.index])
    
    print(df.to_string())
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a FISH-Tuning experiment from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file.")
    args = parser.parse_args()

    # Load base config
    with open('configs/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    # Load experiment-specific config
    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    # Simple deep merge of configs (exp overrides base)
    # A more robust solution would use a dedicated library like OmegaConf
    def merge_configs(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = merge_configs(base[key], value)
            else:
                base[key] = value
        return base

    CFG = merge_configs(base_config, exp_config)
    
    run_generic_experiment(CFG)
