"""
FoNE (Fourier Number Embedding) initialization and override logic.

This module implements the FoNE feature extraction for numbers 0-999 and their
space-prefixed variants, replacing token embeddings with zero-padded frozen FoNE features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


def fone_features(x: int) -> torch.Tensor:
    """
    Compute the 18-dimensional FoNE feature for integer x using multiple frequencies.
    
    Args:
        x: Integer in range [0, 999]
        
    Returns:
        18-dimensional tensor: [cos(2π*x/T), sin(2π*x/T)] for T in [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    """
    # Frequency periods
    periods = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    features = []
    for T in periods:
        # Compute cos(2π*x/T) and sin(2π*x/T)
        angle = 2 * np.pi * x / T
        features.append(np.cos(angle))
        features.append(np.sin(angle))
    
    return torch.tensor(features, dtype=torch.float32)


# Legacy alias for backward compatibility
def fone6(x: int) -> torch.Tensor:
    """Legacy 6D FoNE (deprecated). Use fone_features() instead."""
    return fone_features(x)


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


def pad_fone_features(fone_feats: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """
    Pad 18D FoNE features with zeros to match the hidden dimension.
    
    Args:
        fone_feats: 18-dimensional FoNE features
        hidden_size: Target hidden dimension
        
    Returns:
        Zero-padded features of size hidden_size
    """
    # Get the FoNE feature dimension (18 for the new method)
    fone_dim = fone_feats.shape[0]
    
    # Create zero tensor of target size
    padded = torch.zeros(hidden_size, dtype=fone_feats.dtype, device=fone_feats.device)
    # Copy FoNE features to the beginning
    padded[:fone_dim] = fone_feats
    return padded


def apply_fone_overrides(
    model: nn.Module, 
    tokenizer: PreTrainedTokenizer, 
    hi: int = 999, 
    freeze_rows: bool = True
) -> Dict[str, any]:
    """
    Apply FoNE overrides to model embeddings for numbers 0-hi.
    FoNE features are zero-padded to match hidden dimension and frozen.
    
    Args:
        model: LlamaForCausalLM model
        tokenizer: HuggingFace tokenizer
        hi: Maximum number to override (inclusive, default 999)
        freeze_rows: Whether to freeze the FoNE embedding rows (True for training)
        
    Returns:
        Dict containing override information
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
    
    # Get embedding layer configuration
    d_model = model.config.hidden_size
    embed_tokens = model.get_input_embeddings()
    target_device = embed_tokens.weight.device
    target_dtype = embed_tokens.weight.dtype
    
    logger.info(f"Using zero-padded FoNE features: 18 features (9 frequencies × 2) + {d_model - 18} zeros = {d_model} dimensions")
    
    # Initialize embeddings with zero-padded FoNE features
    overridden_indices = set()
    
    with torch.no_grad():
        for token_str, token_id in found_tokens:
            # Extract the numeric value
            num_str = token_str.strip()
            try:
                num_value = int(num_str)
                if 0 <= num_value <= hi:
                    # Compute FoNE features (18D with multiple frequencies)
                    fone_feats = fone_features(num_value).to(
                        device=target_device,
                        dtype=target_dtype,
                    )
                    
                    # Pad with zeros to match hidden dimension
                    padded_features = pad_fone_features(fone_feats, d_model)
                    
                    # Initialize embedding
                    embed_tokens.weight[token_id] = padded_features
                    overridden_indices.add(token_id)
                    
                    logger.debug(f"Initialize token {token_id} ('{token_str}') with zero-padded FoNE({num_value})")
                    
            except ValueError:
                logger.warning(f"Could not parse number from token: '{token_str}'")
                continue
    
    logger.info(f"Initialized {len(overridden_indices)} embedding rows with zero-padded FoNE features")
    
    # Freeze the number embedding rows if requested
    if freeze_rows and overridden_indices:
        # Create a parameter hook to zero out gradients for frozen rows
        def freeze_number_embeddings_hook(grad):
            """Zero out gradients for number embedding rows."""
            grad_clone = grad.clone()
            for idx in overridden_indices:
                grad_clone[idx] = 0.0
            return grad_clone
        
        # Register the hook on the embedding weight
        embed_tokens.weight.register_hook(freeze_number_embeddings_hook)
        logger.info(f"Froze {len(overridden_indices)} number embedding rows via gradient hook")
        logger.info("Number embeddings will remain fixed during training")
    
    # Return information about the overrides
    return {
        'overridden_indices': overridden_indices,
        'num_overridden': len(overridden_indices),
        'added_tokens': added_tokens,
        'found_tokens': found_tokens,
        'missing_tokens': missing_tokens
    }


def verify_fone_overrides(model: nn.Module, tokenizer: PreTrainedTokenizer, hi: int = 999) -> bool:
    """
    Verify that FoNE overrides have been applied correctly.
    Checks that embeddings match zero-padded FoNE features.
    
    Args:
        model: Model to verify
        tokenizer: Tokenizer to use for verification
        hi: Maximum number that should be overridden
        
    Returns:
        True if verification passes, False otherwise
    """
    # Skip verification if FoNE is disabled (hi < 0 means no number embeddings)
    if hi < 0:
        logger.info("FoNE disabled (hi < 0), skipping verification")
        return True
    
    embed_tokens = model.get_input_embeddings()
    token_info = find_number_tokens(tokenizer, hi)
    d_model = model.config.hidden_size
    
    verification_passed = True
    
    for token_str, token_id in token_info['found']:
        num_str = token_str.strip()
        try:
            num_value = int(num_str)
            if 0 <= num_value <= hi:
                # Compute expected zero-padded FoNE features
                expected_fone = fone_features(num_value).to(
                    device=embed_tokens.weight.device,
                    dtype=embed_tokens.weight.dtype,
                )
                expected_embedding = pad_fone_features(expected_fone, d_model)
                
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


def finalize_fone_embeddings(model: nn.Module, tokenizer: PreTrainedTokenizer, hi: int = 999):
    """
    Finalize FoNE embeddings after training.
    Since embeddings are frozen with zero-padded FoNE features, this is a no-op,
    but kept for compatibility with existing checkpoint code.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        hi: Maximum number for FoNE
    """
    # Skip finalization if FoNE is disabled (baseline model)
    if hi < 0:
        logger.info("FoNE disabled (hi < 0), skipping finalization")
        return
    
    logger.info("FoNE embeddings are already frozen with zero-padded features - no finalization needed")
    
    # Verify embeddings are still correct
    token_info = find_number_tokens(tokenizer, hi)
    found_tokens = token_info['found']
    embed_tokens = model.get_input_embeddings()
    d_model = model.config.hidden_size
    
    verified_count = 0
    with torch.no_grad():
        for token_str, token_id in found_tokens:
            num_str = token_str.strip()
            try:
                num_value = int(num_str)
                if 0 <= num_value <= hi:
                    # Verify embedding is still zero-padded FoNE
                    expected_fone = fone_features(num_value).to(
                        device=embed_tokens.weight.device,
                        dtype=embed_tokens.weight.dtype,
                    )
                    expected_embedding = pad_fone_features(expected_fone, d_model)
                    actual_embedding = embed_tokens.weight[token_id]
                    
                    if torch.allclose(actual_embedding, expected_embedding, atol=1e-5):
                        verified_count += 1
                    else:
                        logger.warning(f"Number embedding {token_id} ('{token_str}') has drifted from expected FoNE value")
                    
            except ValueError:
                continue
    
    logger.info(f"Verified {verified_count}/{len(found_tokens)} FoNE embeddings remain frozen")
