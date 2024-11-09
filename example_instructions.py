from typing import Optional
import os
import file
from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    The main function is to generate responses from the Llama model.

    Parameters:
        ckpt_dir (str): Path to the checkpoint directory for the model weights.
        tokenizer_path (str): Path to the tokenizer file for tokenizing input text.
        Temperature (float): Sampling temperature for text generation (default: 0.2). 
                             Lower values make the output more focused and deterministic.
        top_p (float): Top-p (nucleus) sampling threshold (default: 0.95). Controls diversity of generation.
        max_seq_len (int): Maximum input sequence length (default: 512).
        max_batch_size (int): A maximum number of instructions to process in a batch (default: 8).
        max_gen_len (Optional[int]): Maximum length of the generated output. None means no limit.

    Returns:
        None
    """

    # Check if the checkpoint directory exists
    if not os. Path.dir(ckpt_dir):
        raise ValueError(f"Checkpoint directory '{ckpt_dir}' not found.")
    
    # Check if the tokenizer file exists
    If not os. Path.file(tokenizer_path):
        raise ValueError(f"Tokenizer file '{tokenizer_path}' not found.")

    # Build the generator object from the Llama model with the given parameters
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Example instructions to demonstrate how the generator works
    instructions = [
        [
            {
                "role": "user",
                "content": "In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?",
            }
        ],
        [
            {
                "role": "user",
                "content": "What is the difference between inorder and preorder traversal? Give an example in Python.",
            }
        ],
        [
            {
                "role": "system",
                "content": "Provide answers in JavaScript",
            },
            {
                "role": "user",
                "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
            }
        ],
    ]

    # Generate responses for each instruction with the given parameters
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Display results in a readable format
    for instruction, result in zip(instructions, results):
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
