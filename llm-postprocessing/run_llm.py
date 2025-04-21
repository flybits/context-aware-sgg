# run_llm.py
# run openai models

import argparse
import torch
from llm_utils import read_prompt_from_file, prompt_openai_llm, prompt_huggingface_llm

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Interact with various LLMs using a prompt file.")

    # Argument for specifying the prompt file
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to the text file containing the prompt."
    )

    # Argument for specifying the LLM to use
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        choices=[
            'gpt-4', 'gpt-4o', 'gpt-3.5-turbo',
            'llama2-7b', 'llama2-13b', 'llama2-70b',
            'meta-llama-3.1-8b', 'meta-llama-3.1-70b',
            'meta-llama-3.1-8b-instruct', 'meta-llama-3.1-70b-instruct',
            'falcon-7b', 'falcon-40b',
            'gpt-neox-20b', 'gpt-j-6b'
        ],
        help="The LLM to use."
    )

    # Optional argument for device selection
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help="Device to run the model on (default: cpu)."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Retrieve the prompt file path and LLM choice from the parsed arguments
    prompt_file = args.prompt_file
    llm_choice = args.llm
    device = args.device

    # Read the prompt from the specified file
    prompt = read_prompt_from_file(prompt_file)

    # Generate response based on the chosen LLM
    if llm_choice in ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo']:
        response = prompt_openai_llm(prompt, llm_choice)
    else:

# Map llm_choice to model names on Hugging Face
        model_mapping = {
            'llama2-7b': 'meta-llama/Llama-2-7b-hf',
            'llama2-13b': 'meta-llama/Llama-2-13b-hf',
            'llama2-70b': 'meta-llama/Llama-2-70b-hf',
            'meta-llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B',
            'meta-llama-3.1-70b': 'meta-llama/Meta-Llama-3.1-70B',
            'meta-llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'meta-llama-3.1-70b-instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'meta-llama-3.2-1b-instruct': 'meta-llama/Meta-Llama-3.2-1B-Instruct',
            'meta-llama-3.2-3b-instruct': 'meta-llama/Meta-Llama-3.2-3B-Instruct',
            'falcon-7b': 'tiiuae/falcon-7b-instruct',
            'falcon-40b': 'tiiuae/falcon-40b-instruct',
            'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
            'gpt-j-6b': 'EleutherAI/gpt-j-6B'
        }


        model_name = model_mapping.get(llm_choice)
        if not model_name:
            raise ValueError(f"Model name for {llm_choice} not found.")

        # Check if CUDA is available when device is set to 'cuda'
        if device == 'cuda' and not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Please install CUDA or set --device cpu.")



        response = prompt_huggingface_llm(prompt, model_name, device=device)

    print(response)

if __name__ == "__main__":
    main()
