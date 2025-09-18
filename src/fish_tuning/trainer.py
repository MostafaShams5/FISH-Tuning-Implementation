# src/fish_tuning/trainer.py

import torch
from transformers import Trainer
from typing import Dict

class FishTuningTrainer(Trainer):
    """
    A custom Hugging Face Trainer that applies a pre-computed sparsity mask to gradients.

    This trainer overrides the default `Trainer` to apply a fixed binary mask to the
    gradients of the model's parameters during the backward pass. This is achieved
    efficiently using PyTorch's `register_hook` mechanism.
    """
    def __init__(self, *args, mask: Dict[str, torch.Tensor] = None, **kwargs):
        """
        Initializes the FishTuningTrainer.

        Args:
            *args: Positional arguments passed to the parent `Trainer`.
            mask (Dict[str, torch.Tensor], optional): A dictionary mapping parameter names
                                                      to their binary masks. If None or empty,
                                                      behaves like a standard Trainer.
            **kwargs: Keyword arguments passed to the parent `Trainer`.
        """
        # First, call the constructor of the parent class (`transformers.Trainer`).
        super().__init__(*args, **kwargs)

        # If no mask is provided, there's nothing more to do.
        if mask is None or not mask:
            return

        print("Applying FISH-Tuning gradient mask via hooks...")
        
        # Iterate over all named parameters in the model to attach hooks where needed.
        for name, param in self.model.named_parameters():
            # Check if a mask exists for this specific parameter.
            if name in mask:
                # A PyTorch hook needs to be attached to a parameter that requires gradients.
                if param.requires_grad:
                    # Get the mask tensor for the current parameter.
                    mask_tensor = mask[name]
                    
                    # This is a crucial step for correctness and efficiency. We register the mask
                    # as a "buffer" on the same module that owns the parameter. A buffer is a
                    # tensor that is part of the model's state (and moves to the correct device
                    # with `.to(device)`) but is not considered a trainable parameter.
                    # We get the module and the final parameter name (e.g., 'weight', 'bias').
                    module_name, param_name = name.rsplit('.', 1)
                    module = dict(self.model.named_modules())[module_name]
                    module.register_buffer(f"{param_name}_mask", mask_tensor)

                    # We define a "hook factory" function. This is necessary to correctly
                    # capture the `module` and `param_name` for each parameter in the loop.
                    # Without this factory, all hooks would incorrectly use the last values
                    # of `module` and `param_name` from the loop.
                    def create_mask_grad_hook(module_instance, parameter_name):
                        # This is the actual hook function that PyTorch will call.
                        def hook(grad):
                            # Inside the hook, retrieve the mask that we stored as a buffer.
                            # `getattr` allows us to access the buffer using a dynamic name.
                            buffer_mask = getattr(module_instance, f"{parameter_name}_mask")
                            
                            # The core operation: multiply the gradient by the binary mask.
                            # This effectively zeros out the gradients for pruned parameters,
                            # preventing them from being updated by the optimizer.
                            return grad * buffer_mask
                        return hook
                    
                    # Register the hook on the parameter. The created hook will be automatically
                    # executed by PyTorch's autograd engine right after the gradient for this
                    # specific parameter has been computed.
                    param.register_hook(create_mask_grad_hook(module, param_name))