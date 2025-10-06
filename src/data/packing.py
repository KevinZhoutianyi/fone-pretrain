"""
Efficient sequence packing for language model pretraining.

This module implements efficient packing of variable-length sequences into
fixed-length training examples to maximize GPU utilization.
"""

import torch
from typing import Iterator, Dict, List, Optional, Any
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class SequencePacker:
    """
    Efficient sequence packer that concatenates tokenized sequences and
    cuts them into fixed-length windows for training.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        sequence_length: int = 2048,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        add_eos_to_sequences: bool = True,
        pack_efficiency_threshold: float = 0.8
    ):
        """
        Initialize sequence packer.
        
        Args:
            tokenizer: HuggingFace tokenizer
            sequence_length: Target sequence length for packed examples
            pad_token_id: Token ID for padding (defaults to tokenizer pad token)
            eos_token_id: Token ID for EOS (defaults to tokenizer EOS token)
            add_eos_to_sequences: Whether to add EOS tokens between sequences
            pack_efficiency_threshold: Minimum packing efficiency to log warnings
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_eos_to_sequences = add_eos_to_sequences
        self.pack_efficiency_threshold = pack_efficiency_threshold
        
        # Set token IDs
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
            logger.warning("No pad token found, using EOS token for padding")
            
        if self.eos_token_id is None:
            raise ValueError("No EOS token found in tokenizer")
            
        # Internal state
        self.buffer = []  # Token buffer for packing
        self.total_tokens_processed = 0
        self.total_sequences_processed = 0
        self.total_examples_yielded = 0
        
        logger.info(f"Initialized SequencePacker: seq_len={sequence_length}, "
                   f"pad_id={self.pad_token_id}, eos_id={self.eos_token_id}")
        
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text and return token IDs."""
        # Tokenize without special tokens (we'll add EOS ourselves)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Add EOS token if requested
        if self.add_eos_to_sequences:
            tokens.append(self.eos_token_id)
            
        return tokens
        
    def _pack_from_buffer(self) -> Optional[Dict[str, torch.Tensor]]:
        """Pack tokens from buffer into a training example."""
        if len(self.buffer) < self.sequence_length:
            return None
            
        # Extract exactly sequence_length tokens
        packed_tokens = self.buffer[:self.sequence_length]
        self.buffer = self.buffer[self.sequence_length:]
        
        # Create input_ids and labels (same for causal LM)
        input_ids = torch.tensor(packed_tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # Create attention mask (all 1s since we don't pad within sequences)
        attention_mask = torch.ones_like(input_ids)
        
        self.total_examples_yielded += 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
    def add_text(self, text: str) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Add text to the packer and yield any complete examples.
        
        Args:
            text: Input text to tokenize and pack
            
        Yields:
            Packed training examples
        """
        if not text.strip():
            return
            
        # Tokenize the text
        tokens = self._tokenize_text(text)
        
        if not tokens:
            return
            
        # Add to buffer
        self.buffer.extend(tokens)
        self.total_tokens_processed += len(tokens)
        self.total_sequences_processed += 1
        
        # Yield complete examples
        while len(self.buffer) >= self.sequence_length:
            example = self._pack_from_buffer()
            if example is not None:
                yield example
                
    def finalize(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Finalize packing and yield any remaining examples.
        
        This should be called at the end of the dataset to handle
        remaining tokens in the buffer.
        
        Yields:
            Final packed training examples
        """
        # Pad the buffer to create a final example if we have enough tokens
        if len(self.buffer) > 0:
            if len(self.buffer) >= self.sequence_length * 0.5:  # At least 50% full
                # Pad to sequence length
                padding_needed = self.sequence_length - len(self.buffer)
                self.buffer.extend([self.pad_token_id] * padding_needed)
                
                example = self._pack_from_buffer()
                if example is not None:
                    yield example
            else:
                logger.info(f"Discarding {len(self.buffer)} tokens in final buffer "
                           f"(less than 50% of sequence length)")
                           
    def get_stats(self) -> Dict[str, Any]:
        """Get packing statistics."""
        if self.total_examples_yielded > 0:
            avg_tokens_per_example = self.total_tokens_processed / self.total_examples_yielded
            packing_efficiency = avg_tokens_per_example / self.sequence_length
        else:
            avg_tokens_per_example = 0
            packing_efficiency = 0
            
        return {
            'total_tokens_processed': self.total_tokens_processed,
            'total_sequences_processed': self.total_sequences_processed,
            'total_examples_yielded': self.total_examples_yielded,
            'tokens_in_buffer': len(self.buffer),
            'avg_tokens_per_example': avg_tokens_per_example,
            'packing_efficiency': packing_efficiency,
            'sequence_length': self.sequence_length
        }


class StreamingPackedDataset:
    """
    Streaming dataset that packs sequences from a text iterator.
    """
    
    def __init__(
        self,
        text_iterator: Iterator[str],
        tokenizer: PreTrainedTokenizer,
        sequence_length: int = 2048,
        max_examples: Optional[int] = None,
        **packer_kwargs
    ):
        """
        Initialize streaming packed dataset.
        
        Args:
            text_iterator: Iterator yielding text strings
            tokenizer: HuggingFace tokenizer
            sequence_length: Target sequence length
            max_examples: Maximum number of examples to yield (None for unlimited)
            **packer_kwargs: Additional arguments for SequencePacker
        """
        self.text_iterator = text_iterator
        self.sequence_length = sequence_length
        self.max_examples = max_examples
        
        # Initialize packer
        self.packer = SequencePacker(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            **packer_kwargs
        )
        
        self.examples_yielded = 0
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over packed examples."""
        try:
            for text in self.text_iterator:
                # Check if we've reached the maximum
                if self.max_examples and self.examples_yielded >= self.max_examples:
                    break
                    
                # Pack the text and yield examples
                for example in self.packer.add_text(text):
                    yield example
                    self.examples_yielded += 1
                    
                    if self.max_examples and self.examples_yielded >= self.max_examples:
                        break
                        
        except Exception as e:
            logger.error(f"Error in streaming dataset: {e}")
            raise
        finally:
            # Finalize and yield remaining examples
            if not self.max_examples or self.examples_yielded < self.max_examples:
                for example in self.packer.finalize():
                    yield example
                    self.examples_yielded += 1
                    
                    if self.max_examples and self.examples_yielded >= self.max_examples:
                        break
                        
        # Log final statistics
        stats = self.packer.get_stats()
        logger.info(f"Streaming dataset completed. Stats: {stats}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = self.packer.get_stats()
        stats['examples_yielded'] = self.examples_yielded
        return stats


def create_packed_dataloader(
    text_iterator: Iterator[str],
    tokenizer: PreTrainedTokenizer,
    sequence_length: int = 2048,
    batch_size: int = 1,
    max_examples: Optional[int] = None,
    **packer_kwargs
) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Create a dataloader that yields batched packed examples.
    
    Args:
        text_iterator: Iterator yielding text strings
        tokenizer: HuggingFace tokenizer  
        sequence_length: Target sequence length
        batch_size: Batch size (currently only supports batch_size=1)
        max_examples: Maximum number of examples
        **packer_kwargs: Additional arguments for SequencePacker
        
    Yields:
        Batched packed examples
    """
    if batch_size != 1:
        raise NotImplementedError("Batch size > 1 not yet implemented")
        
    dataset = StreamingPackedDataset(
        text_iterator=text_iterator,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        max_examples=max_examples,
        **packer_kwargs
    )
    
    for example in dataset:
        # Add batch dimension
        batched_example = {
            key: tensor.unsqueeze(0) for key, tensor in example.items()
        }
        yield batched_example


def estimate_packing_efficiency(
    text_samples: List[str],
    tokenizer: PreTrainedTokenizer,
    sequence_length: int = 2048,
    num_samples: int = 1000
) -> float:
    """
    Estimate packing efficiency for a set of text samples.
    
    Args:
        text_samples: List of text samples
        tokenizer: HuggingFace tokenizer
        sequence_length: Target sequence length
        num_samples: Number of samples to use for estimation
        
    Returns:
        Estimated packing efficiency (0.0 to 1.0)
    """
    packer = SequencePacker(tokenizer, sequence_length)
    
    # Process samples
    for i, text in enumerate(text_samples[:num_samples]):
        list(packer.add_text(text))  # Consume the iterator
        
    # Finalize
    list(packer.finalize())
    
    stats = packer.get_stats()
    return stats['packing_efficiency']


if __name__ == "__main__":
    # Example usage and testing
    from transformers import AutoTokenizer
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Test with sample texts
    sample_texts = [
        "This is a short text.",
        "This is a longer text that contains more information and should be tokenized into more tokens.",
        "Here's another example with different content to test the packing functionality.",
        "Short.",
        "A medium length text that falls somewhere in between the short and long examples.",
    ]
    
    # Test packer
    packer = SequencePacker(tokenizer, sequence_length=50)  # Small for testing
    
    print("Testing SequencePacker:")
    example_count = 0
    
    for text in sample_texts:
        print(f"\nAdding text: {text[:50]}...")
        for example in packer.add_text(text):
            example_count += 1
            print(f"Example {example_count}:")
            print(f"  input_ids shape: {example['input_ids'].shape}")
            print(f"  first 10 tokens: {example['input_ids'][:10].tolist()}")
            
    # Finalize
    for example in packer.finalize():
        example_count += 1
        print(f"Final example {example_count}:")
        print(f"  input_ids shape: {example['input_ids'].shape}")
        
    # Print stats
    stats = packer.get_stats()
    print(f"\nPacking statistics: {stats}")
    
    # Test streaming dataset
    print("\nTesting StreamingPackedDataset:")
    
    def text_generator():
        for text in sample_texts * 3:  # Repeat for more data
            yield text
            
    dataset = StreamingPackedDataset(
        text_iterator=text_generator(),
        tokenizer=tokenizer,
        sequence_length=50,
        max_examples=5
    )
    
    for i, example in enumerate(dataset):
        print(f"Streaming example {i+1}: shape {example['input_ids'].shape}")
        
    final_stats = dataset.get_stats()
    print(f"Final streaming stats: {final_stats}")
