import os
import json
import torch
import random
from datasets import DatasetDict, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

def load_and_prepare_data(data_path, eval_size=22):
    """
    Load and prepare the dataset from a single JSONL file.

    Args:
        data_path (str): Path to the data file.
        eval_size (int): Number of examples to use for evaluation.

    Returns:
        DatasetDict: A Hugging Face DatasetDict containing train and eval datasets.
    """
    def load_jsonl(file_path):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]

    data = load_jsonl(data_path)
    
    random.shuffle(data)
    
    # Split the data into train and eval sets
    if eval_size > 0:
        train_data = data[:-eval_size]
        eval_data = data[-eval_size:]
    else:
        train_data = data
        eval_data = []

    # Create Hugging Face Datasets
    train_dataset = Dataset.from_list(train_data)
    if eval_data:
        eval_dataset = Dataset.from_list(eval_data)
    else:
        eval_dataset = None

    dataset = DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset
    })

    return dataset

def tokenize_function(examples, tokenizer):
    """
    Tokenize the input and output text using the provided tokenizer.

    Args:
        examples (dict): A batch of examples from the dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.

    Returns:
        dict: Tokenized inputs.
    """
    # Prepare the prompts
    instructions = examples['instruction']
    inputs = examples['input']
    outputs = examples['output']

    prompts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        prompt = f"{instruction}\n\nInput: {input_text}\n\nResponse: {output_text}"
        prompts.append(prompt)

    # Tokenize the prompts
    tokenized = tokenizer(
        prompts,
        max_length=512,
        truncation=True,
        padding='max_length',
    )

    # Use input_ids as labels
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized



def main():
    # Set environment variables to suppress tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_path = '../data/llama-finetune-train-val.jsonl'
    model_path = 'meta-llama/Llama-3.2-3B-Instruct'

    # Load and prepare the dataset
    dataset = load_and_prepare_data(data_path, eval_size=22)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding token

    # Configure bitsandbytes for 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    # Load the model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map='auto'
    )

    print(f"Model hidden size: {model.config.hidden_size}")

    # Verify model device
    device = next(model.parameters()).device
    print(f"Model is loaded on: {device}")

    # List model modules to identify correct target_modules for LoRA
    print("Listing model modules to identify target_modules for LoRA:")
    for name, module in model.named_modules():
        print(name)

    # Configure PEFT (LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],  # Update based on actual module names
    )
    model = get_peft_model(model, peft_config)

    # Print trainable parameters to verify
    print("Trainable parameters after applying PEFT (LoRA):")
    model.print_trainable_parameters()

    # Tokenize the dataset
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=['instruction', 'input', 'output']
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./llama-finetuned',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size to fit GPU memory
        per_device_eval_batch_size=1,   # Reduced batch size to fit GPU memory
        eval_strategy='steps',          # Use 'eval_strategy' instead of deprecated 'evaluation_strategy'
        eval_steps=500,
        save_steps=1000,
        warmup_steps=100,
        logging_dir='./logs',
        logging_steps=100,
        save_total_limit=2,
        fp16=True,                      # Enable mixed precision if supported
        dataloader_num_workers=4,
        gradient_accumulation_steps=8,
        report_to='none',               # Set to 'wandb' or 'tensorboard' if using
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['eval'] if 'eval' in tokenized_datasets else None,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the model
    model = model_path.split('/')[-1]
    trainer.save_model(f'./llama-finetuned-{model}-model')

if __name__ == "__main__":
    main()
