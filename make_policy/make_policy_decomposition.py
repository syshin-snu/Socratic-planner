import os
import natsort
import json
import argparse
from typing import List, Tuple

import openai

from ..prompt.prompt_decomposition_zeroshot import DecomQAPrompt, PolicyPrompt, PolicyPrompt_noQA


def setup_openai_api():
    """Loads the OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key

def load_instructions(alfred_data_path: str, split: str = 'valid_seen') -> Tuple[List[str], List[str]]:
    """Loads all instructions from the ALFRED dataset."""
    data_path = os.path.join(alfred_data_path, 'json_2.1.0', split)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"ALFRED data directory not found: {data_path}")

    instructions = []
    traj_paths = []
    folder_list = natsort.natsorted(os.listdir(data_path))

    for train_folder in folder_list:
        trials_folder_path = os.path.join(data_path, train_folder)
        trials_folder_list = os.listdir(trials_folder_path)
        for folder in trials_folder_list:
            traj_data_path = os.path.join(trials_folder_path, folder, 'traj_data.json')
            with open(traj_data_path, 'r') as f:
                traj_data = json.load(f)
            for ann in traj_data['turk_annotations']['anns']:
                instructions.append(ann['task_desc'])
                traj_paths.append(traj_data_path)
    return instructions, traj_paths

class PolicyGenerator:
    """A class to generate QA and policies using the OpenAI API."""
    def __init__(self, model="gpt-3.5-turbo", temperature=0):
        self.model = model
        self.temperature = temperature

    def _call_api(self, messages: List[dict], max_tokens: int) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message["content"]

    def generate_qa(self, instruction: str, max_tokens: int = 500) -> str:
        messages = [{"role": "user", "content": DecomQAPrompt(instruction)}]
        return self._call_api(messages, max_tokens)

    def generate_policy_with_qa(self, instruction: str, qa_result: str, max_tokens: int = 300) -> str:
        messages = [{"role": "user", "content": PolicyPrompt(instruction, qa_result)}]
        return self._call_api(messages, max_tokens)

    def generate_policy_without_qa(self, instruction: str, max_tokens: int = 300) -> str:
        messages = [{"role": "user", "content": PolicyPrompt_noQA(instruction)}]
        return self._call_api(messages, max_tokens)

def save_results(filepath: str, results_list: List[Tuple[str, str]]):
    """Saves a list of (instruction, result) tuples to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for instruction, result in results_list:
                f.write(f"Instruction: {instruction}\n\n")
                f.write(f"{result}\n\n---\n\n")
        print(f"Results saved to {filepath}.")
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")

def main(args: argparse.Namespace):
    """Main execution function."""
    setup_openai_api()
    instructions, traj_paths = load_instructions(args.alfred_data_path, args.split)
    print(f"Loaded {len(instructions)} instructions.")

    generator = PolicyGenerator(model=args.model, temperature=args.temperature)

    qa_results = []
    policy_with_qa_results = []
    policy_without_qa_results = []
    error_instructions = []

    for i, inst in enumerate(instructions):
        print(f"\n----- Processing instruction {i+1}/{len(instructions)}: {inst} -----")
        try:
            qa_result = generator.generate_qa(inst, max_tokens=args.max_tokens_qa)
            print("\n--- Generated QA ---")
            print(qa_result)
            qa_results.append((inst, qa_result))

            policy_with_qa = generator.generate_policy_with_qa(inst, qa_result, max_tokens=args.max_tokens_policy)
            print("\n--- Policy with QA ---")
            print(policy_with_qa)
            policy_with_qa_results.append((inst, policy_with_qa))

            if args.generate_no_qa_policy:
                policy_without_qa = generator.generate_policy_without_qa(inst, max_tokens=args.max_tokens_policy)
                print("\n--- Policy without QA ---")
                print(policy_without_qa)
                policy_without_qa_results.append((inst, policy_without_qa))

        except Exception as e:
            print(f"Error occurred: {inst} - {e}")
            error_instructions.append(inst)

    # Save results
    save_results(args.qa_output, qa_results)
    save_results(args.policy_output, policy_with_qa_results)
    if args.generate_no_qa_policy:
        save_results(args.no_qa_policy_output, policy_without_qa_results)

    if error_instructions:
        with open(args.error_log, 'w', encoding='utf-8') as f:
            for inst in error_instructions:
                f.write(f"{inst}\n")
        print(f"Saved {len(error_instructions)} instructions that caused errors to {args.error_log}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose and generate policies using the ALFRED dataset.")
    parser.add_argument('--alfred_data_path', type=str, required=True, help='Root path of the ALFRED dataset.')
    parser.add_argument('--split', type=str, default='valid_seen', help='Data split to use (e.g., valid_seen, valid_unseen).')
    parser.add_argument('--qa_output', type=str, default='decomposed_qa.txt', help='File path to save the generated QA.')
    parser.add_argument('--policy_output', type=str, default='decomposed_policy.txt', help='File path to save the policy based on QA.')
    parser.add_argument('--no_qa_policy_output', type=str, default='decomposed_policy_no_qa.txt', help='File path to save the policy without QA.')
    parser.add_argument('--error_log', type=str, default='error_log.txt', help='File to log instructions that caused an error.')
    parser.add_argument('--generate_no_qa_policy', action='store_true', help='Whether to generate policies without QA.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='OpenAI model to use.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature to use for generation.')
    parser.add_argument('--max_tokens_qa', type=int, default=500, help='Max tokens for QA generation.')
    parser.add_argument('--max_tokens_policy', type=int, default=300, help='Max tokens for policy generation.')

    args = parser.parse_args()
    main(args)