# scripts/run_experiment.py (Definitive Final Version with Helper Functions Restored)

import torch
import numpy as np
import pandas as pd
import yaml
import os
import time
from datasets import load_dataset
from sklearn.metrics import accuracy_score # Use reliable, local scikit-learn for accuracy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
import sys

# --- Robust Import of Local Module ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from fish_tuning import (
    FishTuningTrainer,
    calculate_fisher_information,
    create_mask_from_fisher,
)

# --- START: HELPER FUNCTIONS (RESTORED) ---
def count_trainable_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_keep_ratio(baseline_params, target_params):
    """Calculates the keep_ratio needed to match the baseline parameter count."""
    return baseline_params / target_params
# --- END: HELPER FUNCTIONS (RESTORED) ---


# --- Resource Tracking Utilities ---
def get_gpu_memory_usage():
    """Returns peak memory usage in MB using PyTorch's official tracker."""
    if not torch.cuda.is_available():
        return "N/A"
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    return f"{peak_mem_bytes / 1024**2:.2f} MB"

class CudaTimer:
    """A context manager for timing CUDA operations accurately."""
    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

    def elapsed_seconds(self):
        return self.elapsed_time_ms / 1000

# =================================================================================
# MAIN EXPERIMENT RUNNER
# =================================================================================
def run_generic_experiment(config):
    cfg_model = config['model']
    cfg_dataset = config['dataset']
    cfg_lora = config['lora']
    cfg_training = config['training']
    cfg_fish = config['fish_tuning']

    # --- Load Dataset and Tokenizer ---
    print(f"Loading dataset '{cfg_dataset['name']}' and tokenizer for '{cfg_model['name']}'...")
    dataset = load_dataset(*cfg_dataset['load_args'])
    tokenizer = AutoTokenizer.from_pretrained(cfg_model['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        cfg_model.setdefault('config_overrides', {})['pad_token_id'] = tokenizer.eos_token_id

    # --- Define Metric Computation ---
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}

    def preprocess_function(examples):
        return tokenizer(examples[cfg_dataset['text_column']], truncation=True, padding='max_length', max_length=cfg_training['max_seq_length'])
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    results = {}
    num_labels = len(dataset['train'].unique(cfg_dataset.get('label_column', 'label')))

    # =============================================================================
    # EXPERIMENT A: BASELINE (ORIGINAL LORA)
    # =============================================================================
    print("\n" + "="*50)
    print("RUNNING EXPERIMENT A: BASELINE (ORIGINAL LORA)")
    print("="*50)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    with CudaTimer() as timer:
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
        
        training_args_dict = {**config['base_training']['args'], **cfg_training.get('args_override', {})}
        training_args = TrainingArguments(
            output_dir=os.path.join(cfg_training['output_dir'], "baseline_lora"), **training_args_dict
        )
        trainer_baseline = Trainer(
            model=model_baseline_peft, args=training_args, train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset[cfg_dataset['validation_split']], tokenizer=tokenizer,
            data_collator=data_collator, compute_metrics=compute_metrics,
        )
        trainer_baseline.train()
    
    peak_mem = get_gpu_memory_usage()
    eval_key = f"eval_{training_args.metric_for_best_model}"
    eval_results_baseline = trainer_baseline.evaluate()
    
    results['Original LoRA'] = {
        'Train Time (s)': f"{timer.elapsed_seconds():.2f}",
        'Peak GPU Mem': peak_mem,
        'Final Trainable Params': baseline_trainable_params,
        f'Val {training_args.metric_for_best_model.capitalize()}': eval_results_baseline[eval_key]
    }
    
    del model_baseline, model_baseline_peft, trainer_baseline
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # =============================================================================
    # SETUP & RUN FOR MASKED METHODS
    # =============================================================================
    if cfg_fish.get('methods_to_run'):
        print("\n" + "="*50)
        print("PRE-COMPUTATION FOR MASKED METHODS")
        print("="*50)
        
        with CudaTimer() as timer_precomp:
            model_for_fish = AutoModelForSequenceClassification.from_pretrained(
                cfg_model['name'], num_labels=num_labels, **cfg_model.get('config_overrides', {})
            )
            lora_config_fish = LoraConfig(
                r=cfg_lora['fish_rank'], lora_alpha=2 * cfg_lora['fish_rank'],
                target_modules=cfg_lora['target_modules'], lora_dropout=cfg_lora['dropout'],
                bias="none", task_type=TaskType.SEQ_CLS,
            )
            model_for_fish_peft = get_peft_model(model_for_fish, lora_config_fish)
            
            keep_ratio = calculate_keep_ratio(baseline_trainable_params, count_trainable_parameters(model_for_fish_peft))
            
            calibration_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(cfg_fish['num_samples']))
            cols_to_remove = [cfg_dataset['text_column']]
            if 'idx' in calibration_dataset.column_names: cols_to_remove.append('idx')
            calibration_dataset = calibration_dataset.remove_columns(cols_to_remove)
            calibration_dataloader = DataLoader(calibration_dataset, batch_size=1, collate_fn=data_collator)

            fisher_scores = {}
            if "fish" in cfg_fish['methods_to_run'] or "rev" in cfg_fish['methods_to_run']:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                fisher_scores = calculate_fisher_information(
                    model=model_for_fish_peft, calibration_dataloader=calibration_dataloader, device=device
                )

            names_to_exclude = cfg_lora.get('names_to_exclude', [])
            all_masks = {
                "fish": create_mask_from_fisher(fisher_scores, keep_ratio, names_to_exclude=names_to_exclude),
                "rand": create_mask_from_fisher({name: torch.rand_like(p) for name, p in model_for_fish_peft.named_parameters() if p.requires_grad}, keep_ratio, names_to_exclude=names_to_exclude),
                "rev": create_mask_from_fisher({name: -s for name, s in fisher_scores.items()}, keep_ratio, names_to_exclude=names_to_exclude),
            }
        
        print(f"Pre-computation (Fisher, masks) took: {timer_precomp.elapsed_seconds():.2f} seconds.")
        del model_for_fish, model_for_fish_peft
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        for method_key in cfg_fish['methods_to_run']:
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            with CudaTimer() as timer_masked:
                exp_name = {"fish": "LoRA-FISH (Ours)", "rand": "LoRA-FISH-rand", "rev": "LoRA-FISH-rev"}[method_key]
                mask = all_masks[method_key]
                
                print("\n" + "="*50)
                print(f"RUNNING EXPERIMENT: {exp_name}")
                print("="*50)
                
                model_to_train = AutoModelForSequenceClassification.from_pretrained(
                    cfg_model['name'], num_labels=num_labels, **cfg_model.get('config_overrides', {})
                )
                model_to_train_peft = get_peft_model(model_to_train, lora_config_fish)
                
                masked_training_args = TrainingArguments(
                    output_dir=os.path.join(cfg_training['output_dir'], exp_name.lower().replace(" ", "_").replace("(", "").replace(")", "")),
                    **training_args_dict
                )
                trainer_masked = FishTuningTrainer(
                    model=model_to_train_peft, args=masked_training_args, mask=mask,
                    train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset[cfg_dataset['validation_split']],
                    tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics,
                )
                trainer_masked.train()
            
            peak_mem_masked = get_gpu_memory_usage()
            eval_key_masked = f"eval_{masked_training_args.metric_for_best_model}"
            eval_results_masked = trainer_masked.evaluate()
            
            results[exp_name] = {
                'Train Time (s)': f"{timer_masked.elapsed_seconds():.2f}",
                'Peak GPU Mem': peak_mem_masked,
                'Final Trainable Params': baseline_trainable_params,
                f'Val {masked_training_args.metric_for_best_model.capitalize()}': eval_results_masked[eval_key_masked]
            }
            
            del model_to_train, model_to_train_peft, trainer_masked
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # =============================================================================
    # PRINT FINAL RESULTS
    # =============================================================================
    print("\n" + "="*80)
    print(f"          FINAL EXPERIMENT RESULTS for {cfg_model['name']} on {cfg_dataset['name']}")
    print("="*80)
    
    df = pd.DataFrame.from_dict(results, orient='index')
    metric_col_name = f'Val {training_args.metric_for_best_model.capitalize()}'
    if metric_col_name in df.columns:
        df[metric_col_name] = df[metric_col_name].apply(lambda x: f"{x:.4f}")
    
    desired_order = ['Original LoRA', 'LoRA-FISH (Ours)', 'LoRA-FISH-rand', 'LoRA-FISH-rev']
    df = df.reindex([idx for idx in desired_order if idx in df.index])
    
    print(df.to_string())
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a FISH-Tuning experiment from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file.")
    args = parser.parse_args()

    with open('configs/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    def merge_configs(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = merge_configs(base[key], value)
            else:
                base[key] = value
        return base

    CFG = merge_configs(base_config, exp_config)
    run_generic_experiment(CFG)
