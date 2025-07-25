import os
import datetime

# --- Patched to use default paths if environment variables are not set ---
# This avoids needing to source init.sh or export variables for every session.
# The default workspace is set to '~/socratic_planner_workspace'.
DEFAULT_WORKSPACE_DIR = os.path.expanduser("~/socratic_planner_workspace")

ROOT_PATH = os.environ.get("LGP_WS_DIR", DEFAULT_WORKSPACE_DIR)
MODEL_PATH = os.environ.get("LGP_MODEL_DIR", os.path.join(ROOT_PATH, "models"))
DATA_PATH = os.environ.get("LGP_DATA_DIR", os.path.join(ROOT_PATH, "data"))

# Ensure the necessary directories exist.
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

print(f"INFO: LGP_WS_DIR set to: {ROOT_PATH}")
print(f"INFO: LGP_MODEL_DIR set to: {MODEL_PATH}")
print(f"INFO: LGP_DATA_DIR set to: {DATA_PATH}")


def get_root_dir():
    return ROOT_PATH


def get_model_dir():
    return MODEL_PATH


def get_checkpoint_dir():
    return os.path.join(get_root_dir(), "checkpoints")


# Root for all data
def get_data_dir():
    os.makedirs(DATA_PATH, exist_ok=True)
    return DATA_PATH


# Root for rollouts collected from the ALFRED environment
def get_default_rollout_data_dir():
    return os.path.join(get_data_dir(), "rollouts")


# Root for rollouts processed for subgoal model training
def get_default_subgoal_rollout_data_dir():
    return os.path.join(get_default_rollout_data_dir(), "alfred_subgoal_rollouts")


# Root for rollouts processed for navigation and perception model training
def get_default_navigation_rollout_data_dir():
    return os.path.join(get_default_rollout_data_dir(), "alfred_navigation_data_full")


def get_artifact_output_path():
    pth = os.path.join(ROOT_PATH, "data", "artifacts")
    os.makedirs(pth, exist_ok=True)
    return pth


def get_experiment_runs_dir(exp_name):
    rundir = os.path.join(ROOT_PATH, "data", "runs", exp_name)
    return rundir


def get_results_dir(exp_name):
    resultsdir = os.path.join(ROOT_PATH, "data", "results", exp_name)
    return resultsdir


def get_leaderboard_progress_path(exp_name):
    leaderboard_file_path = os.path.join(get_results_dir(exp_name), "leaderboard_progress.json")
    return leaderboard_file_path


def get_leaderboard_result_path(exp_name):
    leaderboard_dir = get_results_dir(exp_name)
    leaderboard_file = os.path.join(leaderboard_dir, 'tests_actseqs_dump_' + datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f") + '.json')
    return leaderboard_file


def get_eval_rollout_dir(exp_name):
    rdir = os.path.join(get_results_dir(exp_name), "rollouts")
    os.makedirs(rdir, exist_ok=True)
    return rdir


def get_depth_model_path():
    import lgp.parameters
    exp_def = lgp.parameters.get_experiment_definition()
    depth_file = exp_def.Setup.agent_setup.depth_model_file
    pth = os.path.join(MODEL_PATH, depth_file)
    return pth


def get_segmentation_model_path():
    import lgp.parameters
    exp_def = lgp.parameters.get_experiment_definition()
    seg_file = exp_def.Setup.agent_setup.seg_model_file
    pth = os.path.join(MODEL_PATH, seg_file)
    return pth


def get_navigation_model_path():
    import lgp.parameters
    exp_def = lgp.parameters.get_experiment_definition()
    nav_file = exp_def.Setup.agent_setup.navigation_model_file
    pth = os.path.join(MODEL_PATH, nav_file)
    return pth


def get_subgoal_model_path():
    import lgp.parameters
    exp_def = lgp.parameters.get_experiment_definition()
    nav_file = exp_def.Setup.agent_setup.subgoal_model_file
    pth = os.path.join(MODEL_PATH, nav_file)
    return pth
