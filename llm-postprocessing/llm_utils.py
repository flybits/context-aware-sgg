# llm_utils.py

import os
import torch
from peft import PeftModel
from openai import OpenAI, OpenAIError
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import infer_auto_device_map
import torch



def read_prompt_from_file(file_path):
    """
    Reads the prompt from the specified text file.

    Args:
        file_path (str): Path to the text file containing the prompt.

    Returns:
        str: The content of the prompt file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Prompt file '{file_path}' does not exist.")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()
            if not prompt:
                raise ValueError(f"Prompt file '{file_path}' is empty.")
            return prompt
    except Exception as e:
        raise SystemExit(f"Error reading prompt file '{file_path}': {e}")

def prompt_openai_llm(prompt, model):
    """
    Sends a prompt to the OpenAI GPT model and returns the response.

    Args:
        prompt (str): The prompt to send to the GPT model.
        model (str): The GPT model to use (e.g., "gpt-4").

    Returns:
        str: The response from the GPT model.
    """
    # Initialize the OpenAI client with the API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    client = OpenAI(
        api_key=api_key,
    )

    try:
        # Create the chat completion request
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )
    except OpenAIError as e:
        raise SystemExit(f"OpenAI API returned an error: {e}")

    # Extract and return the response text
    llm_output = chat_completion.choices[0].message.content.strip()
    return llm_output


# def prompt_huggingface_llm(prompt, model_path, device='cuda'):
#     """
#     Sends a prompt to a Hugging Face LLM model and returns the response.

#     Args:
#         prompt (str): The prompt to send to the LLM.
#         model_path (str): Path to the local model directory or model name on Hugging Face.
#         device (str): The device to run the model on ('cpu' or 'cuda').

#     Returns:
#         str: The response from the LLM.
#     """
#     try:

#         # Load the tokenizer and model
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
        
#         if device == 'cuda':
#             model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
#         else:
#             model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

#         # If the model is an instruct model, format the prompt accordingly
#         if 'Instruct' in model_path:
#             system_prompt = "You are a helpful assistant."
#             prompt = f"""[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"""

#         # Prepare inputs
#         inputs = tokenizer(prompt, return_tensors='pt').to(device)

#         # Generate response
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=256,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             num_return_sequences=1,
#             pad_token_id=tokenizer.eos_token_id
#         )

#         # Decode and return the response
#         generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # Remove the prompt from the generated text
#         response = generated_text[len(prompt):].strip()
#         return response
#     except Exception as e:
#         raise SystemExit(f"Error generating response from the model: {e}")



def prompt_huggingface_llm(prompt, model_path, device='cuda'):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=True)

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        # Load the model with quantization and device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map='auto',
            token=True
        )

        # If the model is an instruct model, format the prompt accordingly
        if 'Instruct' in model_path:
            system_prompt = "You are an assistant tasked with converting input tuples from the format (subject, predicate, object) to the format (situation, attribute, value). Closely follow the instructions, and ensure each tuple is accurately transformed without adding extra information."
            prompt = f"""[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"""

        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # Generate response with adjusted parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode and return the response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        return response

    except Exception as e:
        raise SystemExit(f"Error generating response from the model: {e}")

def get_openai_finetuned_response(input_text, model_name):
    client = OpenAI()
    completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "system", "content": "You are an assistant tasked with converting tuples from Form A to Form B."},

        {"role": "user", "content": input_text}
    ],
    temperature=0
    )
    return completion.choices[0].message

# Function to read the .txt test file and return the list of test cases
def read_test_txt_file(file_path):
    test_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip to remove any leading/trailing whitespaces including newline characters
            test_list.append(line.strip())
    return test_list


def create_base_prompt(prompt_path):
    with open(prompt_path, 'r') as file:
        base_prompt = file.read()
    return base_prompt


def load_meta_finetuned_model(model_dir, base_model_name):
    """
    Load the fine-tuned model with PEFT (LoRA) adapters.

    Args:
        model_dir (str): Path to the fine-tuned model directory.
        base_model_name (str): Name of the base model.

    Returns:
        model: The loaded model ready for inference.
        tokenizer: The corresponding tokenizer.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding token

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map='auto',
        load_in_8bit=True,  # Load in 8-bit if you used bitsandbytes
        torch_dtype=torch.float16  # Use float16 for faster inference
    )

    # Load the PEFT (LoRA) adapters
    model = PeftModel.from_pretrained(
        base_model,
        model_dir
    )

    # Set the model to evaluation mode
    model.eval()

    return model, tokenizer

def get_meta_finetuned_response(model, tokenizer, prompt, max_length=4098, temperature=0.0001, top_p=0.9):
    """
    Generate a response from the model based on the input prompt.

    Args:
        model: The loaded model.
        tokenizer: The corresponding tokenizer.
        prompt (str): The input prompt for the model.
        max_length (int): Maximum length of the generated response.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.

    Returns:
        str: The generated response.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response