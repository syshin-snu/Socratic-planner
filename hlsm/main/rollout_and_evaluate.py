"""
Script that rolls out an agent and does not much else for now
"""
import sys
import os

# -- Add project root to sys.path --
# This is necessary to resolve the ModuleNotFoundError for 'lgp'.
# The lgp module is located in the parent directory of 'main'.
script_dir = os.path.dirname(os.path.abspath(__file__))
hlsm_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(hlsm_dir)
sys.path.insert(0, hlsm_dir)
sys.path.insert(0, project_root)

# -- Add ALFRED source to sys.path --
# This is necessary to resolve the ModuleNotFoundError for 'alfred'.
# It assumes 'alfred_src' is at the project root, parallel to 'hlsm'.
# The ALFRED project itself uses non-relative imports, so we need to add
# both the parent 'alfred_src' and the 'alfred' directory to the path.
alfred_src_path = os.path.join(project_root, 'alfred_src')
sys.path.insert(0, alfred_src_path)
sys.path.insert(0, os.path.join(alfred_src_path, 'alfred'))

import torch
import numpy as np
import argparse
import json

from lgp.agents.agents import get_agent
from lgp.rollout.rollout_actor import RolloutActorLocal
from lgp.metrics.alfred_eval import get_multiple_rollout_metrics_alfred
from main.visualize_rollout import visualize_rollout
from lgp.parameters import Hyperparams, load_experiment_definition

from main.eval_progress import EvalProgress

from lgp.env.alfred.alfred_env import AlfredEnv

def evaluate_rollouts(exp_def, rollouts):
    metrics = get_multiple_rollout_metrics_alfred(rollouts)
    print("Results: ")
    metrics.printout()
    return metrics


def main(args):
    # params
    exp_def = Hyperparams(load_experiment_definition(args.def_name))
    device = torch.device(exp_def.Setup.device)
    dataset_device = torch.device(exp_def.Setup.dataset_device)
    exp_name = exp_def.Setup.experiment_name
    horizon = exp_def.Setup.horizon
    num_rollouts = exp_def.Setup.num_rollouts
    visualize_rollouts = exp_def.Setup.visualize_rollouts
    save_animation_dir = exp_def.Setup.get("save_rollout_animations_dir", False)

    env = AlfredEnv(device=device,
                    setup=exp_def.Setup.env_setup.d,
                    hparams=exp_def.Hyperparams.d)

    agent = get_agent(exp_def.Setup, exp_def.Hyperparams, device)

    rollout_actor = RolloutActorLocal(experiment_name=exp_name,
                                      agent=agent,
                                      env=env,
                                      dataset_proc=None,
                                      param_server_proc=None,
                                      max_horizon=horizon,
                                      dataset_device=dataset_device,
                                      index=1,
                                      collect_trace=visualize_rollouts,
                                      lightweight_mode=not visualize_rollouts,
                                      do_replan=args.replan)

    # Track progress
    eval_progress = EvalProgress(exp_name)

    # Load policies and QA data from paths provided as arguments
    if not os.path.exists(args.policies_path):
        raise FileNotFoundError(f"Policies file not found at: {args.policies_path}")
    with open(args.policies_path, 'r') as file:
        policies = json.load(file)

    if not os.path.exists(args.qa_path):
        print(f"Warning: QA file not found at: {args.qa_path}. Replanning might fail.")

    # Collect the rollouts
    for i in range(num_rollouts):
        print(f"Rollout {i}/{num_rollouts}")
        try:
            # The call to rollout is simplified. Logging is now handled inside the actor.
            rollout = rollout_actor.rollout(i, policies, args.qa_path, skip_tasks=eval_progress.get_done_tasks())

            if rollout is not None:
                eval_progress.add_rollout(rollout)
                eval_progress.save()

                if save_animation_dir is not None:
                    for s in rollout:
                        if s["observation"] is not None:
                            s["observation"].compress()
                    os.makedirs(save_animation_dir, exist_ok=True)
                    visualize_rollout(rollout, save_animation_dir, f"rollout_{i}", start_t=0)

        except Exception as e:
            print(f"An error occurred during rollout {i}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Export the leaderboard file
    eval_progress.export_leaderboard_json()

    # Get numbers for table on validation sets
    metrics = evaluate_rollouts(exp_def, eval_progress.get_rollout_list())

    # Save final metrics to a file
    if args.results_path:
        with open(args.results_path, 'w') as f:
            f.write(metrics.to_json())
        print(f"Final metrics saved to {args.results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout and evaluate a Socratic Planner agent.")
    parser.add_argument("def_name", type=str, help="Experiment definition file name (e.g., alfred/eval/hlsm_full/eval_hlsm_valid_seen)")
    parser.add_argument("--policies_path", type=str, required=True, help="Path to the JSON file containing pre-computed policies.")
    parser.add_argument("--qa_path", type=str, required=True, help="Path to the QA file used for replanning.")
    parser.add_argument("--results_path", type=str, default="evaluation_results.json", help="Path to save the final evaluation metrics.")
    parser.add_argument("--replan", action="store_true", help="Enable replanning during rollout.")

    args = parser.parse_args()
    main(args)