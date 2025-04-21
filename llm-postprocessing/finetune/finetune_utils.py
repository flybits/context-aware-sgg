import csv
import ast
import json

# Function to filter tuples based on keywords
def filter_tuples(data, keywords):
    filtered_data = []
    for row in data:
        # Parse the string representation of the list of tuples into an actual list of tuples
        tuples = ast.literal_eval(row[1])
        # Filter out tuples containing any of the keywords
        filtered_tuples = [t for t in tuples if not any(kw in t for kw in keywords)]
        # Update the row with the filtered list of tuples
        if filtered_tuples:
            filtered_data.append((row[0], filtered_tuples))
    return filtered_data

# Read CSV file and process the data
def read_and_filter_csv(file_path, keywords):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header
        data = list(csv_reader)

    # Filter tuples based on keywords
    filtered_data = filter_tuples(data, keywords)
    return filtered_data

def preprocess_openai_finetune(X_data, Y_data, output_path, eval_size=22):
    X = [item[-1] for item in X_data]
    Y = [item[-1] for item in Y_data]

    ## Saving samples for unbiased evaluation
    print(X[-eval_size:])
    assert len(X) == len(Y), "The number of inputs and outputs must be the same."

    data = []
    for inp, out in zip(X, Y):
        item = {
            "messages": [
                {"role": "user", "content": f"{inp}"},
                {"role": "assistant", "content": f"{out}"}
            ]
        }
        data.append(item)

    train_val_path = output_path[:-6] + "-train-val.jsonl"
    unbiased_eval_path = output_path[:-6] + "-eval.jsonl"

    with open(train_val_path, 'w') as f:
        for item in data[:-eval_size]:
            json.dump(item, f)
            f.write('\n')

    print(f"Train/validation data has been prepared and written to {train_val_path}")

    with open(unbiased_eval_path, 'w') as f:
        for item in data[-eval_size:]:
            json.dump(item, f)
            f.write('\n')

    print(f"Evaluation data has been prepared and written to {unbiased_eval_path}")

    return None


def preprocess_llama_finetune(X_data, Y_data, output_path, eval_size=22):
    X = [item[-1] for item in X_data]
    Y = [item[-1] for item in Y_data]

    assert len(X) == len(Y), "The number of inputs and outputs must be the same."

    data = []
    for inp, out in zip(X, Y):
        instruction = "You are a helpful assistant."
        # instruction = "You are an assistant tasked with converting tuples from Form A to Form B."
        input_text = f"{inp}"
        output_text = f"{out}"
        example = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        data.append(example)

    # Split data into training/validation and evaluation
    train_val_data = data[:-eval_size]
    eval_data = data[-eval_size:]

    train_val_path = output_path[:-6] + "-train-val.jsonl"
    unbiased_eval_path = output_path[:-6] + "-eval.jsonl"

    with open(train_val_path, 'w') as f:
        for item in train_val_data:
            json.dump(item, f)
            f.write('\n')

    with open(unbiased_eval_path, 'w') as f:
        for item in eval_data:
            json.dump(item, f)
            f.write('\n')

    print(f"Train/validation data has been prepared and written to {train_val_path}")
    print(f"Evaluation data has been prepared and written to {unbiased_eval_path}")

    return None

