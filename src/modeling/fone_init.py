"""
FoNE (Fourier Number Embedding) initialization and override logic.

This module implements the FoNE feature extraction for numbers 0-999 and their
space-prefixed variants, replacing token embeddings with frozen FoNE features
passed through a learnable linear projection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


def fone6(x: int) -> torch.Tensor:
    """
    Compute the 6-dimensional FoNE feature for integer x.
    
    Args:
        x: Integer in range [0, 999]
        
    Returns:
        6-dimensional tensor: [cos(x0/10), sin(x0/10), cos(x1/10), sin(x1/10), cos(x2/10), sin(x2/10)]
        where x0, x1, x2 are units, tens, hundreds digits respectively.
    """
    # Zero-pad to 3 digits and extract digits
    x_str = f"{x:03d}"
    x2 = int(x_str[0])  # hundreds
    x1 = int(x_str[1])  # tens  
    x0 = int(x_str[2])  # units
    
    # Compute FoNE features (using radians as specified, no 2π)
    features = torch.tensor([
        np.cos(x0 / 10.0),  # units cos
        np.sin(x0 / 10.0),  # units sin
        np.cos(x1 / 10.0),  # tens cos
        np.sin(x1 / 10.0),  # tens sin
        np.cos(x2 / 10.0),  # hundreds cos
        np.sin(x2 / 10.0),  # hundreds sin
    ], dtype=torch.float32)
    
    return features


def find_number_tokens(tokenizer: PreTrainedTokenizer, hi: int = 999) -> Dict[str, List[int]]:
    """
    Find token IDs for numbers 0-hi. Only looks for single number tokens, not space-prefixed.
    
    Args:
        tokenizer: HuggingFace tokenizer
        hi: Maximum number to search for (inclusive)
        
    Returns:
        Dict with 'found' and 'missing' keys containing lists of (number, token_id) tuples
    """
    found = []
    missing = []
    
    for num in range(hi + 1):
        num_str = str(num)
        
        # Check regular number token
        try:
            tokens = tokenizer.encode(num_str, add_special_tokens=False)
            if len(tokens) == 1:
                found.append((num_str, tokens[0]))
            else:
                missing.append(num_str)
        except:
            missing.append(num_str)
    
    return {
        'found': found,
        'missing': missing
    }


def add_missing_number_tokens(tokenizer: PreTrainedTokenizer, missing_tokens: List[str]) -> List[str]:
    """
    Add missing number tokens to the tokenizer vocabulary.
    
    Args:
        tokenizer: HuggingFace tokenizer to modify
        missing_tokens: List of token strings to add
        
    Returns:
        List of successfully added tokens
    """
    if not missing_tokens:
        return []
        
    logger.info(f"Adding {len(missing_tokens)} missing number tokens to vocabulary")
    
    # Filter out tokens that might already exist
    tokens_to_add = []
    for token in missing_tokens:
        try:
            # Try encoding - if it results in a single token, it might already exist
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                tokens_to_add.append(token)
        except:
            tokens_to_add.append(token)
    
    if tokens_to_add:
        added_tokens = tokenizer.add_tokens(tokens_to_add)
        logger.info(f"Successfully added {added_tokens} new tokens")
        return tokens_to_add[:added_tokens]
    
    return []


class FrozenEmbeddingHook:
    """Hook to freeze specific embedding rows during training."""
    
    def __init__(self, frozen_indices: Set[int]):
        self.frozen_indices = frozen_indices
        
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """Zero out gradients for frozen embedding rows."""
        if grad is not None:
            grad[list(self.frozen_indices)] = 0.0
        return grad


def apply_fone_overrides(
    model: nn.Module, 
    tokenizer: PreTrainedTokenizer, 
    hi: int = 999, 
    freeze_rows: bool = True
) -> Dict[str, any]:
    """
    Apply FoNE overrides to model embeddings for numbers 0-hi.
    
    Args:
        model: LlamaForCausalLM model
        tokenizer: HuggingFace tokenizer
        hi: Maximum number to override (inclusive, default 999)
        freeze_rows: Whether to freeze the overridden embedding rows
        
    Returns:
        Dict containing override information and fone_proj module
    """
    logger.info(f"Applying FoNE overrides for numbers 0-{hi}")
    
    # Find existing number tokens
    token_info = find_number_tokens(tokenizer, hi)
    found_tokens = token_info['found']
    missing_tokens = token_info['missing']
    
    logger.info(f"Found {len(found_tokens)} existing number tokens")
    logger.info(f"Missing {len(missing_tokens)} number tokens")
    
    # Add missing tokens if any
    added_tokens = []
    if missing_tokens:
        added_tokens = add_missing_number_tokens(tokenizer, missing_tokens)
        if added_tokens:
            # Resize model embeddings to accommodate new tokens
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")
            
            # Re-scan for tokens after adding
            token_info = find_number_tokens(tokenizer, hi)
            found_tokens = token_info['found']
    
    # Create or get the FoNE projection layer (match embedding dtype/device)
    d_model = model.config.hidden_size
    embed_tokens = model.get_input_embeddings()
    target_device = embed_tokens.weight.device
    target_dtype = embed_tokens.weight.dtype
    if not hasattr(model, 'fone_proj'):
        proj = nn.Linear(6, d_model, bias=False)
        proj = proj.to(device=target_device, dtype=target_dtype)
        model.fone_proj = proj
        logger.info(f"Created FoNE projection layer: 6 -> {d_model}")
    
    # Apply overrides to embedding matrix
    overridden_indices = set()
    
    with torch.no_grad():
        for token_str, token_id in found_tokens:
            # Extract the numeric value
            num_str = token_str.strip()
            try:
                num_value = int(num_str)
                if 0 <= num_value <= hi:
                    # Compute FoNE features
                    fone_features = fone6(num_value).to(
                        device=embed_tokens.weight.device,
                        dtype=embed_tokens.weight.dtype,
                    )
                    
                    # Project through learnable layer (stays in model dtype)
                    projected = model.fone_proj(fone_features)
                    
                    # Override embedding
                    embed_tokens.weight[token_id] = projected
                    overridden_indices.add(token_id)
                    
                    logger.debug(f"Override token {token_id} ('{token_str}') with FoNE({num_value})")
                    
            except ValueError:
                logger.warning(f"Could not parse number from token: '{token_str}'")
                continue
    
    logger.info(f"Applied FoNE overrides to {len(overridden_indices)} embedding rows")
    
    # Set up freezing hook if requested
    frozen_hook = None
    if freeze_rows and overridden_indices:
        frozen_hook = FrozenEmbeddingHook(overridden_indices)
        embed_tokens.weight.register_hook(frozen_hook)
        logger.info(f"Registered freezing hook for {len(overridden_indices)} embedding rows")
    
    # Return information about the overrides
    return {
        'overridden_indices': overridden_indices,
        'num_overridden': len(overridden_indices),
        'added_tokens': added_tokens,
        'fone_proj': model.fone_proj,
        'frozen_hook': frozen_hook,
        'found_tokens': found_tokens,
        'missing_tokens': missing_tokens
    }


def verify_fone_overrides(model: nn.Module, tokenizer: PreTrainedTokenizer, hi: int = 999) -> bool:
    """
    Verify that FoNE overrides have been applied correctly.
    
    Args:
        model: Model to verify
        tokenizer: Tokenizer to use for verification
        hi: Maximum number that should be overridden
        
    Returns:
        True if verification passes, False otherwise
    """
    if not hasattr(model, 'fone_proj'):
        logger.error("Model does not have fone_proj layer")
        return False
        
    embed_tokens = model.get_input_embeddings()
    token_info = find_number_tokens(tokenizer, hi)
    
    verification_passed = True
    
    for token_str, token_id in token_info['found']:
        num_str = token_str.strip()
        try:
            num_value = int(num_str)
            if 0 <= num_value <= hi:
                # Compute expected FoNE features
                expected_fone = fone6(num_value).to(
                    device=embed_tokens.weight.device,
                    dtype=embed_tokens.weight.dtype,
                )
                expected_embedding = model.fone_proj(expected_fone)
                
                # Compare with actual embedding (cast to float32 for stable comparison)
                actual_embedding = embed_tokens.weight[token_id]
                if not torch.allclose(
                    actual_embedding.float(), expected_embedding.float(), atol=1e-5
                ):
                    logger.error(f"FoNE override verification failed for token {token_id} ('{token_str}')")
                    verification_passed = False
                    
        except ValueError:
            continue
    
    if verification_passed:
        logger.info("FoNE override verification passed")
    else:
        logger.error("FoNE override verification failed")
        
    return verification_passed
