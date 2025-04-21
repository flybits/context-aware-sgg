import csv
import ast
import json
from finetune_utils import filter_tuples, read_and_filter_csv, preprocess_openai_finetune, preprocess_llama_finetune


"""
This file prepares scene graph and situation graph data in the format expected for finetuning with LLM apis.
"""


def main():
    ## X inputs (generated scene graphs from sgg models)
    X_file_path = '../finetune-raw-data/motifs-tde-boxtopk20-reltopk20-output.csv'  
    ## Y Ground Truths (labelled graphs with desired situation graph output)
    Y_file_path = '../finetune-raw-data/labels-v1.2.csv'

    ## Remove certain types of classes that are impossible to infer from scene graph data
    X_keywords_to_remove = [] # No need to remove from input
    Y_keywords_to_remove = ['has time', 'has participant', 'has emotion', 'main participant', 'family and friends', 'friends', 'unknown']


    X_filtered = read_and_filter_csv(X_file_path, X_keywords_to_remove)
    Y_filtered = read_and_filter_csv(Y_file_path, Y_keywords_to_remove)
    print(X_filtered, Y_filtered)

    # for item in filtered_inputs:
    #     print(item[-1])


    # preprocess_openai_finetune(filtered_X, filtered_Y, output_path='openai-finetune.jsonl', eval_size=22)
    preprocess_llama_finetune(X_filtered, Y_filtered, output_path='llama-finetune.jsonl', eval_size=22)

if __name__ == '__main__':
    main()