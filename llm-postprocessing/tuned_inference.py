import json
from openai import OpenAI
from llm_utils import get_openai_finetuned_response, read_test_txt_file, create_base_prompt, get_meta_finetuned_response, load_meta_finetuned_model


## OPENAI MODELS INFERENCE
# client = OpenAI()

# Fine-tuned model name
# model_name = 'INSERT-MODEL-NAME-HERE'


## Single inference
# input_message = "[('hand', 'holding', 'glass'), ('bottle', 'in', 'hand'), ('hand', 'holding', 'bottle'), ('glass', 'in', 'hand'), ('building', 'behind', 'woman'), ('man', 'has', 'head'), ('building', 'behind', 'man'), ('man', 'has', 'head'), ('man', 'has', 'hand'), ('man', 'holding', 'bottle'), ('building', 'behind', 'person'), ('man', 'holding', 'glass'), ('person', 'wearing', 'shirt'), ('hand', 'holding', 'glass'), ('man', 'has', 'head'), ('glass', 'in', 'hand'), ('person', 'holding', 'bottle'), ('person', 'has', 'hand'), ('woman', 'wearing', 'shirt'), ('woman', 'has', 'hand')]"

# Get and print the response
# output = get_openai_finetuned_response(input_message)
# print("Assistant's Response:")
# print(output)

# Define the test path
# test_path = 'finetune/test-x.txt'

# Read the file and get the list of test cases
# test_list = read_test_txt_file(test_path)

# curr_index = 1


# for test_sample in test_list:
#     input_message = base_prompt + test_sample
#     print(f"SGG INPUT on img {curr_index}: ", test_sample, '\n')
#     output = get_openai_finetuned_response(input_message, model_name)
#     print(f"OUTPUT on img {curr_index}: ", output.content, '\n')
#     curr_index+=1





## META MODELS INFERENCE
## Paths
model_dir = 'finetune/meta/llama-finetuned-Llama-3.2-3B-Instruct-model' 
base_model_name = 'meta-llama/Llama-3.2-3B-Instruct'
test_path = 'finetune/test-x.txt'

base_prompt_path = 'rule-based-prompt.txt'
base_prompt = create_base_prompt(base_prompt_path)

# Load the model and tokenizer
model, tokenizer = load_meta_finetuned_model(model_dir, base_model_name)
print("Model and tokenizer loaded successfully.")

test_list = read_test_txt_file(test_path)
curr_index = 1

for test_sample in test_list:
    input_message = base_prompt + test_sample
    print(f"SGG INPUT on img {curr_index}: ", test_sample, '\n')
    output = get_meta_finetuned_response(model, tokenizer, input_message)
    print(f"OUTPUT on img {curr_index}: ", output, '\n')
    curr_index+=1
