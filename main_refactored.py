import argparse
import logging
import os
import re
import sys
import random
import numpy as np
import torch
from utils import *

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    args = parse_arguments()
    logging.info(f'Parsed arguments: {args}')

    fix_seed(args.random_seed)

    openai_key = os.getenv("OPENAI_API_KEY")
    logging.info(f"OPENAI_API_KEY: ***")

    # Initialize decoder class (load model and tokenizer)
    decoder = Decoder(args)
    logging.info("Decoder initialized.")

    # Setup data loader
    dataloader = setup_data_loader(args)
    print_now()  # Assuming this is a time logging utility

    demo = setup_demo_text(args) if "few_shot" in args.method else ""
    accuracy = process_data(dataloader, args, decoder, demo)
    logging.info(f"Final accuracy: {accuracy:.2f}%")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--api_log_file_name", type=str, default=None,
                        help="mandatory argument! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]")

    parser.add_argument("--random_seed", type=int,
                        default=1, help="random seed")

    parser.add_argument("--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa",
                        "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment")

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[
                        1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=3,
                        help="maximum number of workers for dataloader")

    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="model used for reasoning and extracting answer")

    parser.add_argument("--method", type=str, default="zero_shot_cot", choices=[
                        "zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"], help="method")

    parser.add_argument("--cot_trigger_no", type=int, default=1,
                        help="A trigger sentence that elicits a model to execute chain of thought")

    parser.add_argument("--max_length_cot", type=int, default=128,
                        help="maximum length of output tokens by model for reasoning extraction")

    parser.add_argument("--max_length_direct", type=int, default=32,
                        help="maximum length of output tokens by model for answer extraction")

    parser.add_argument("--limit_dataset_size", type=int, default=10,
                        help="whether to limit test dataset size. If 0, use all samples in the dataset.")

    parser.add_argument("--api_time_interval", type=float,
                        default=1.0, help="Time interval between API calls")

    parser.add_argument("--log_dir", type=str,
                        default="./log/", help="log directory")

    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(f"Argument parsing failed with error: {e}")
        parser.print_help()
        sys.exit(1)

    configure_dataset(args)
    configure_cot_trigger(args)

    return args


def configure_dataset(args):
    dataset_paths = {
        "aqua": ("./dataset/AQuA/test.json", "\nTherefore, among A through E, the answer is"),
        "gsm8k": ("./dataset/grade-school-math/test.jsonl", "\nTherefore, the answer (arabic numerals) is"),
        "commonsensqa": ("./dataset/CommonsenseQA/dev_rand_split.jsonl", "\nTherefore, among A through E, the answer is"),
        "addsub": ("./dataset/AddSub/AddSub.json", "\nTherefore, the answer (arabic numerals) is"),
        # Add other dataset paths and triggers here
    }

    if args.dataset not in dataset_paths:
        raise ValueError("Dataset is not properly defined.")

    args.dataset_path, args.direct_answer_trigger = dataset_paths[args.dataset]
    args.direct_answer_trigger_for_zeroshot = args.direct_answer_trigger.replace(
        "\nTherefore, ", "").capitalize()
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"


def configure_cot_trigger(args):
    cot_triggers = [
        "Let's think step by step.",
        "We should think about this step by step.",
        "First,",
        "Before we dive into the answer,",
        "Proof followed by the answer.",
        "Let's think step by step in a realistic way.",
        # Add other trigger phrases as needed
    ]

    if 1 <= args.cot_trigger_no <= len(cot_triggers):
        args.cot_trigger = cot_triggers[args.cot_trigger_no - 1]
    else:
        raise ValueError("cot_trigger_no is not properly defined.")


def setup_demo_text(args):
    cot_flag = "cot" in args.method
    return create_demo_text(args, cot_flag=cot_flag)


def process_data(dataloader, args, decoder, demo):
    total, correct_list = 0, []
    for i, data in enumerate(dataloader):
        logging.info(f"Processing {i+1}st data")

        x, y = data
        x = f"Q: {x[0]}\nA:"
        y = y[0].strip()

        # Modify question template based on method
        if args.method == "zero_shot":
            x += f" {args.direct_answer_trigger_for_zeroshot}"
        elif args.method == "zero_shot_cot":
            x += f" {args.cot_trigger}"
        elif "few_shot" in args.method:
            x = demo + x
        else:
            raise ValueError("Method is not properly defined.")

        # Predict answer
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        pred = decoder.decode(args, x, max_length, i, 1)

        # Handle zero-shot-cot secondary prediction
        if args.method == "zero_shot_cot":
            z2 = f"{x}{pred} {args.direct_answer_trigger_for_zeroshot_cot}"
            max_length = args.max_length_direct
            pred = decoder.decode(args, z2, max_length, i, 2)

        # Clean and evaluate prediction
        pred = answer_cleansing(args, pred)
        logging.info(f"Prediction: {pred}, Ground Truth: {y}")

        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1

        if args.limit_dataset_size and (i + 1) >= args.limit_dataset_size:
            break

    # Calculate accuracy
    accuracy = (sum(correct_list) / total) * 100 if total > 0 else 0
    return accuracy


if __name__ == "__main__":
    main()
