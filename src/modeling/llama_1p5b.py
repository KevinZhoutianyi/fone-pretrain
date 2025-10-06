"""
Llama 1.5B model wrapper with FoNE integration and FlashAttention-2 support.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import (
    LlamaConfig, 
    LlamaForCausalLM, 
    AutoTokenizer,
    PreTrainedTokenizer
)
from typing import Optional, Dict, Any
import logging

from .fone_init import apply_fone_overrides, verify_fone_overrides

logger = logging.getLogger(__name__)


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict


def create_llama_model(
    config_path: str,
    tokenizer: PreTrainedTokenizer,
    flash_attention: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16
) -> LlamaForCausalLM:
    """
    Create a Llama model from configuration with optional FlashAttention-2.
    
    Args:
        config_path: Path to model configuration JSON
        tokenizer: Tokenizer to determine vocabulary size
        flash_attention: Whether to use FlashAttention-2
        torch_dtype: Model dtype
        
    Returns:
        Initialized LlamaForCausalLM model
    """
    # Load configuration
    config_dict = load_model_config(config_path)
    
    # Set vocabulary size from tokenizer
    config_dict['vocab_size'] = len(tokenizer)
    
    # Configure FlashAttention-2 if requested
    if flash_attention:
        config_dict['_attn_implementation'] = 'flash_attention_2'
        logger.info("Enabled FlashAttention-2")
    
    # Create Llama configuration
    config = LlamaConfig(**config_dict)
    
    # Create model
    logger.info(f"Creating Llama model with {config.num_hidden_layers} layers, "
                f"{config.hidden_size} hidden size, {config.vocab_size} vocab size")
    
    model = LlamaForCausalLM(config)
    
    # Set model dtype
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
        logger.info(f"Set model dtype to {torch_dtype}")
    
    return model


def setup_llama_with_fone(
    config_path: str,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    flash_attention: bool = False,
    fone_hi: int = 999,
    freeze_fone_rows: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer, Dict[str, Any]]:
    """
    Set up Llama model with FoNE overrides.
    
    Args:
        config_path: Path to model configuration JSON
        tokenizer_name: Name or path of tokenizer to use
        flash_attention: Whether to use FlashAttention-2
        fone_hi: Maximum number for FoNE overrides (0 to fone_hi inclusive)
        freeze_fone_rows: Whether to freeze FoNE embedding rows during training
        torch_dtype: Model dtype
        
    Returns:
        Tuple of (model, tokenizer, fone_info)
    """
    logger.info(f"Setting up Llama model with FoNE overrides (0-{fone_hi})")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Ensure tokenizer has required special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Create model
    model = create_llama_model(
        config_path=config_path,
        tokenizer=tokenizer, 
        flash_attention=flash_attention,
        torch_dtype=torch_dtype
    )
    
    # Apply FoNE overrides
    logger.info("Applying FoNE overrides...")
    fone_info = apply_fone_overrides(
        model=model,
        tokenizer=tokenizer,
        hi=fone_hi,
        freeze_rows=freeze_fone_rows
    )
    
    # Verify overrides
    if not verify_fone_overrides(model, tokenizer, fone_hi):
        logger.warning("FoNE override verification failed!")
    
    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  FoNE overridden embeddings: {fone_info['num_overridden']}")
    logger.info(f"  Added tokens: {len(fone_info['added_tokens'])}")
    
    return model, tokenizer, fone_info


class LlamaWithFoNE(nn.Module):
    """
    Wrapper class for Llama model with FoNE integration.
    
    This class provides a convenient interface for working with FoNE-enhanced
    Llama models, including proper initialization and checkpoint handling.
    """
    
    def __init__(
        self,
        config_path: str,
        tokenizer_name: str = "meta-llama/Llama-3.1-8B", 
        flash_attention: bool = False,
        fone_hi: int = 999,
        freeze_fone_rows: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        
        self.config_path = config_path
        self.tokenizer_name = tokenizer_name
        self.flash_attention = flash_attention
        self.fone_hi = fone_hi
        self.freeze_fone_rows = freeze_fone_rows
        self.torch_dtype = torch_dtype
        
        # Initialize model and tokenizer
        self.model, self.tokenizer, self.fone_info = setup_llama_with_fone(
            config_path=config_path,
            tokenizer_name=tokenizer_name,
            flash_attention=flash_attention,
            fone_hi=fone_hi,
            freeze_fone_rows=freeze_fone_rows,
            torch_dtype=torch_dtype
        )
        
    def forward(self, *args, **kwargs):
        """Forward pass through the underlying model."""
        return self.model(*args, **kwargs)
        
    def generate(self, *args, **kwargs):
        """Generate text using the underlying model."""
        return self.model.generate(*args, **kwargs)
        
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model and tokenizer."""
        # Save the underlying model
        self.model.save_pretrained(save_directory, **kwargs)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save FoNE configuration
        fone_config = {
            'fone_hi': self.fone_hi,
            'freeze_fone_rows': self.freeze_fone_rows,
            'overridden_indices': list(self.fone_info['overridden_indices']),
            'added_tokens': self.fone_info['added_tokens']
        }
        
        with open(os.path.join(save_directory, 'fone_config.json'), 'w') as f:
            json.dump(fone_config, f, indent=2)
            
        logger.info(f"Saved model, tokenizer, and FoNE config to {save_directory}")
        
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        flash_attention: bool = None,
        torch_dtype: torch.dtype = None,
        **kwargs
    ):
        """Load a pre-trained FoNE model."""
        # Load FoNE configuration if it exists
        fone_config_path = os.path.join(model_path, 'fone_config.json')
        if os.path.exists(fone_config_path):
            with open(fone_config_path, 'r') as f:
                fone_config = json.load(f)
                
            fone_hi = fone_config.get('fone_hi', 999)
            freeze_fone_rows = fone_config.get('freeze_fone_rows', True)
        else:
            logger.warning(f"No FoNE config found at {fone_config_path}, using defaults")
            fone_hi = 999
            freeze_fone_rows = True
        
        # Load the base model
        model = LlamaForCausalLM.from_pretrained(model_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Apply FoNE overrides to the loaded model
        fone_info = apply_fone_overrides(
            model=model,
            tokenizer=tokenizer,
            hi=fone_hi,
            freeze_rows=freeze_fone_rows
        )
        
        # Create wrapper instance
        instance = cls.__new__(cls)
        super(LlamaWithFoNE, instance).__init__()
        
        instance.model = model
        instance.tokenizer = tokenizer
        instance.fone_info = fone_info
        instance.fone_hi = fone_hi
        instance.freeze_fone_rows = freeze_fone_rows
        
        if flash_attention is not None:
            instance.flash_attention = flash_attention
        if torch_dtype is not None:
            instance.torch_dtype = torch_dtype
            instance.model = instance.model.to(dtype=torch_dtype)
            
        return instance
        
    def get_memory_footprint(self) -> Dict[str, int]:
        """Get model memory footprint information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate memory usage (rough approximation)
        if self.torch_dtype == torch.bfloat16:
            bytes_per_param = 2
        elif self.torch_dtype == torch.float32:
            bytes_per_param = 4
        else:
            bytes_per_param = 2  # default assumption
            
        model_memory_mb = (total_params * bytes_per_param) / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'estimated_model_memory_mb': int(model_memory_mb),
            'fone_overridden_embeddings': self.fone_info['num_overridden']
        }


def setup_flash_attention():
    """Set up FlashAttention-2 environment variables."""
    if os.getenv('FLASH_ATTENTION') == '1':
        logger.info("FlashAttention-2 enabled via environment variable")
        return True
    return False
