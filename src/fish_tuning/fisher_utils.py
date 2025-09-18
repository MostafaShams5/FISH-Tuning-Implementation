# src/fish_tuning/fisher_utils.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
import re

# calculate_fisher_information function remains the same, as it correctly
# calculates scores for all trainable parameters. The filtering happens next.
def calculate_fisher_information(
    model: torch.nn.Module, 
    calibration_dataloader: DataLoader,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Calculates the diagonal of the Empirical Fisher Information Matrix for trainable parameters.
    ... (rest of the function is unchanged, comments omitted for brevity) ...
    """
    fisher_scores = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_scores[name] = torch.zeros_like(param, device=device)

    model.eval()
    model.to(device)

    for batch in tqdm(calibration_dataloader, desc="Calculating Fisher Information"):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_scores[name] += param.grad.detach().pow(2)
    
    model.train()
    return fisher_scores


# --- THIS FUNCTION IS UPDATED ---
def create_mask_from_fisher(
    fisher_scores: Dict[str, torch.Tensor],
    keep_ratio: float,
    names_to_exclude: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Creates a binary mask from Fisher information scores, keeping the top parameters
    while excluding specified layers from masking.

    Args:
        fisher_scores (Dict[str, torch.Tensor]): The Fisher scores from `calculate_fisher_information`.
        keep_ratio (float): The fraction of parameters to keep *within the sparsified pool*.
        names_to_exclude (Optional[List[str]]): A list of regex patterns. Parameters whose names
                                                match any of these patterns will be fully trained
                                                (i.e., they will get a mask of all ones). A common
                                                example is ['classifier', 'score'].

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping each trainable parameter's name to its binary mask.
    """
    if keep_ratio >= 1.0:
        return {}
    
    # Initialize the dictionary to hold the final binary masks.
    masks = {}
    # Initialize a list to hold the scores of parameters that are eligible for sparsification.
    pool_to_sparsify = []

    # If no exclusion patterns are provided, initialize an empty list.
    if names_to_exclude is None:
        names_to_exclude = []

    # First pass: Separate parameters into two groups: those to be sparsified and those to be excluded (trained densely).
    for name, scores in fisher_scores.items():
        # Check if the parameter name matches any of the exclusion patterns.
        is_excluded = any(re.search(pattern, name) for pattern in names_to_exclude)
        
        if is_excluded:
            # If excluded, create a mask of all ones for this parameter, ensuring it's fully trained.
            masks[name] = torch.ones_like(scores, dtype=torch.float)
        else:
            # If not excluded, add its scores to the pool that will be subject to pruning.
            pool_to_sparsify.append(scores.flatten())

    # If the pool of parameters to sparsify is empty, we are done.
    if not pool_to_sparsify:
        return masks

    # Concatenate all scores in the sparsification pool into a single tensor for global thresholding.
    all_scores_to_sparsify = torch.cat(pool_to_sparsify)

    # Calculate the number of parameters to keep based on the keep_ratio.
    # IMPORTANT: This ratio is applied *only* to the pool of parameters eligible for sparsification.
    num_params_to_keep = int(len(all_scores_to_sparsify) * keep_ratio)

    # If for some reason we end up with zero parameters to keep, we should handle it gracefully.
    if num_params_to_keep == 0:
        # We'll keep at least one parameter to avoid errors, though this is an edge case.
        num_params_to_keep = 1
        print(f"Warning: keep_ratio ({keep_ratio}) resulted in 0 parameters to keep. Keeping 1 instead.")

    # Find the global threshold value from the pool of scores.
    threshold = torch.topk(all_scores_to_sparsify, num_params_to_keep, largest=True).values[-1]
    
    # Second pass: Create the binary masks for the parameters in the sparsification pool.
    for name, scores in fisher_scores.items():
        # Only process parameters that were not excluded in the first pass.
        if name not in masks:
            # Create a binary mask based on the global threshold.
            masks[name] = (scores >= threshold).float()

    return masks