import ray
import torch
import logging
from lgp.abcd.agent import TrainableAgent
from lgp import paths
from lgp.env.alfred.segmentation_definitions import OBJECT_STR_TO_INT, OBJECT_INT_TO_STR, OBJECT_CLASSES
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal

from make_policy.replanning_policy_decomposition import Replanner

MAX_REPLAN_ATTEMPTS = 5
ALFRED_ACTION_LIST = ['PickupObject', 'PutObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject', 'OpenObject', 'CloseObject', 'Stop']


class RolloutActorLocal:

    def __init__(self,
                 experiment_name: str,
                 agent : TrainableAgent,
                 env,
                 dataset_proc,
                 param_server_proc,
                 max_horizon,
                 dataset_device,
                 index,
                 collect_trace=False,
                 lightweight_mode=False,
                 do_replan=False):
        # Setup logging
        self.logger = logging.getLogger(f"RolloutActor_{index}")
        if not self.logger.handlers:
            # Add a handler if not already configured (e.g., by a central logging setup)
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

        self.param_server_proc = param_server_proc
        self.actor_index = index
        if self.actor_index == 0:
            from lgp.utils.better_summary_writer import BetterSummaryWriter
            self.writer = BetterSummaryWriter(f"{paths.get_experiment_runs_dir(experiment_name)}-rollout", start_iter=0)
        else:
            self.writer = None

        self.agent = agent
        self.env = env
        self.horizon = max_horizon
        self.env.set_horizon(max_horizon)
        self.counter = 0
        self.do_replan = do_replan

        self.collect_trace = collect_trace       # Whether to eval outputs of agent.get_trace in the rollout
        self.lightweight_mode = lightweight_mode # Whether to produce stripped-down rollouts with task and metadata only

        self.dataset_device = dataset_device


    def _load_agent_state_from_ps(self):
        for model in self.agent.get_learnable_models():
            model.load_state_dict(ray.get(self.param_server_proc.get.remote(model.get_name())))

    def rollout_and_send_forever(self):
        while True:
            self.rollout_and_send()

    def split_rollout(self, skip_tasks=None, max_section=20, ret=None):
        rollout = []
        with torch.no_grad():
            if ret is None:
                observation, task, rollout_idx = self.env.reset(skip_tasks=skip_tasks)
                # Skipped:
                if task is None:
                    return None, None, True

                print("Task: ", str(task))
                self.agent.start_new_rollout(task)
                action = self.agent.act(observation)
                start = 0
            else:
                observation = ret["observation"]
                action = ret["action"]
                task = ret["task"]
                rollout_idx = ret["rollout_idx"]
                start = ret["t"]

            total_reward = 0
            for t in range(start, self.horizon):
                next_observation, reward, done, md = self.env.step(action)
                total_reward += reward

                rollout.append({
                    "task": task,
                    "observation": None if self.lightweight_mode else (
                        observation.to(self.dataset_device)),
                    "action": None if self.lightweight_mode else action,
                    "reward": reward,
                    "return": total_reward,
                    "agent_trace": self.agent.get_trace(device=self.dataset_device) if (
                            self.collect_trace and not self.lightweight_mode) else None,
                    "done": done,
                    "md": md
                })

                self.agent.clear_trace()

                observation = next_observation

                if done:
                    self.agent.finalize(total_reward)
                    rollout.append({
                        "task": task,
                        "observation": None if self.lightweight_mode else next_observation.to(self.dataset_device),
                        "action": None,
                        "agent_trace": None,
                        "reward": 0,
                        "return": total_reward,
                        "done": True,
                        "md": md  # TODO: This gets added twice, which might be confusing
                    })
                    new_ret = None
                    break
                else:
                    action = self.agent.act(observation)

                if t - start > max_section:
                    new_ret = {
                        "t": t,
                        "task": task,
                        "rollout_idx": rollout_idx,
                        "observation": observation,
                        "action": action
                    }
                    break
                else:
                    new_ret = None

            if new_ret is not None:
                print(f"Pause rollout: {self.counter}, length: {len(rollout)}")
                return rollout, new_ret, False
            else:
                print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                self.counter += 1
                return rollout, new_ret, True
    
    def rollout(self, i, policies, qa_file, skip_tasks=None):
        end_at_start = False
        rollout = []
        with torch.no_grad():
            observation, task, rollout_idx = self.env.reset(skip_tasks=skip_tasks)
            if task is None:
                return None

            self.logger.info(f"Starting rollout for task: {task}")

            self.agent.start_new_rollout(task)

            now_policy = policies[str(i)]

            idx = 0

            if len(now_policy) == 0:
                if self.do_replan:
                    now_policy = self.replan_policies(self.env.thor_env.last_event.frame[:, :, ::-1], str(task), qa_file, now_policy, '', '', OBJECT_CLASSES, idx)
                    if len(now_policy) == 0:
                        print(f"Finished rollout: {self.counter}, length: 0")
                        self.counter += 1
                        return None
                else:
                    print(f"Finished rollout: {self.counter}, length: 0")
                    self.counter += 1
                    return None

            act_type = now_policy[idx]['action']
            arg_type = now_policy[idx]['object']

            try:
                arg_id = OBJECT_STR_TO_INT[arg_type]
            except:
                if self.do_replan:
                    is_end = True
                    for _ in range(MAX_REPLAN_ATTEMPTS):
                        now_policy = self.replan_policies(self.env.thor_env.last_event.frame[:, :, ::-1], str(task), qa_file, now_policy, act_type, arg_type, OBJECT_CLASSES, idx)
                        if len(now_policy) == 0 or len(now_policy) < 1:
                            self.logger.warning("Policy length invalid after replanning!")
                            print(f"Finished rollout: {self.counter}, length: 0")
                            self.counter += 1
                            return None
                        act_type = now_policy[idx]['action']
                        arg_type = now_policy[idx]['object']
                        try:
                            arg_id = OBJECT_STR_TO_INT[arg_type]
                            if act_type not in ALFRED_ACTION_LIST:
                                continue
                            is_end = False
                            break
                        except:
                            continue
                    if is_end:
                        end_at_start = True
                        act_type = "Stop"
                        arg_id = -1
                else:
                    end_at_start = True
                    act_type = "Stop"
                    arg_id = -1

            hl_action = AlfredSubgoal.from_type_str_and_arg_id(act_type, arg_id)
            hl_action = hl_action.to('cuda')

            action = self.agent.act(observation, hl_action)

            total_reward = 0
            ll_action_list = []
            fail_count = 0
            
            seen_items = []
            for t in range(self.horizon):
                visible_items = []
                #print(f"Taking action: {action}")
                next_observation, reward, done, md = self.env.step(action)

                total_reward += reward

                rollout.append({
                    "task": task,
                    "observation": None if self.lightweight_mode else (
                        observation.to(self.dataset_device)),
                    "action": None if self.lightweight_mode else action,
                    "reward": reward,
                    "return": total_reward,
                    "agent_trace": self.agent.get_trace(device=self.dataset_device) if (
                            self.collect_trace and not self.lightweight_mode) else None,
                    "done": done,
                    "md": md
                })

                if end_at_start:
                    print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                    self.counter += 1
                    return rollout

                self.agent.clear_trace()

                past_observation = observation

                observation = next_observation

                custom_failed = False
                rgb_diff = (observation.rgb_image - past_observation.rgb_image).float().abs().mean()
                if rgb_diff < 1e-4:
                    custom_failed = True

                for o in range(123):
                    if observation.semantic_image[0, o, :, :].sum() > 0:
                        seen_items.append(OBJECT_INT_TO_STR[o])
                        visible_items.append(OBJECT_INT_TO_STR[o])


                if done:
                    self.agent.finalize(total_reward)
                    rollout.append({
                        "task": task,
                        "observation": None if self.lightweight_mode else next_observation.to(self.dataset_device),
                        "action": None,
                        "agent_trace": None,
                        "reward": 0,
                        "return": total_reward,
                        "done": True,
                        "md": md # TODO: This gets added twice, which might be confusing
                    })
                    
                    break
                else:
                    ### action = self.agent.act(observation)
                    if str(action).replace('AA: ', "") in ['PickupObject', 'PutObject', 'SliceObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', "ToggleObjectOff"]:
                        self.logger.info(f"{hl_action} : {'X' if custom_failed else 'O'}")

                    action = self.agent.act(observation, hl_action)

                    failed = self.agent.is_failed

                    if custom_failed:
                        fail_count += 1

                    if len(ll_action_list) > 0 and act_type == ll_action_list[-1] and not custom_failed:
                        idx += 1
                        try:
                            act_type = now_policy[idx]['action']
                            arg_type = now_policy[idx]['object']

                            if act_type not in ALFRED_ACTION_LIST:
                                is_in = False
                            else:
                                is_in = True
                            
                            is_in_count = 0
                            while not is_in:
                                if is_in_count == MAX_REPLAN_ATTEMPTS:
                                    print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                                    self.counter += 1
                                    return rollout

                                seen_items_ = list(set(seen_items))
                                now_policy = self.replan_policies(self.env.thor_env.last_event.frame[:, :, ::-1], str(task), qa_file, now_policy, act_type, arg_type, seen_items_, idx)
                                act_type = now_policy[idx]['action']
                                arg_type = now_policy[idx]['object']

                                if act_type not in ALFRED_ACTION_LIST:
                                    is_in = False
                                    is_in_count += 1
                                else:
                                    is_in = True


                            try:
                                arg_id = OBJECT_STR_TO_INT[arg_type]
                            except:
                                if self.do_replan:
                                    is_end = True
                                    for _ in range(MAX_REPLAN_ATTEMPTS):
                                        seen_items_ = list(set(seen_items))
                                        old_policy = now_policy
                                        now_policy = self.replan_policies(self.env.thor_env.last_event.frame[:, :, ::-1], str(task), qa_file, now_policy, act_type, arg_type, seen_items_, idx)
                                        if len(now_policy) == 0 or len(now_policy) <= idx:
                                            self.logger.warning("Policy length invalid after replanning!")
                                            print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                                            self.counter += 1
                                            return rollout
                                        act_type = now_policy[idx]['action']
                                        arg_type = now_policy[idx]['object']
                                        try:
                                            arg_id = OBJECT_STR_TO_INT[arg_type]
                                            if act_type not in ALFRED_ACTION_LIST:
                                                continue
                                            is_end = False
                                            break
                                        except:
                                            continue
                                    if is_end:
                                        print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                                        self.counter += 1
                                        return rollout
                                else:
                                    print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                                    self.counter += 1
                                    return rollout
                        except:  # for the case when the policy does not end with stop sign
                            act_type = 'Stop'
                            arg_id = -1    
                        
                        hl_action = AlfredSubgoal.from_type_str_and_arg_id(act_type, arg_id)
                        hl_action = hl_action.to('cuda')

                        ll_action_list = []
                        fail_count = 0
    
                    else:
                        if fail_count == 4 and self.do_replan:
                            # idx -= 1
                            is_end = True
                            for _ in range(MAX_REPLAN_ATTEMPTS):
                                seen_items_ = list(set(seen_items))
                                now_policy = self.replan_policies(self.env.thor_env.last_event.frame[:, :, ::-1], str(task), qa_file, now_policy, act_type, arg_type, seen_items_, idx)
                                if len(now_policy) == 0 or len(now_policy) <= idx:
                                    self.logger.warning("Policy length invalid after replanning!")
                                    print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                                    self.counter += 1
                                    return rollout
                                act_type = now_policy[idx]['action']
                                arg_type = now_policy[idx]['object']
                                try:
                                    arg_id = OBJECT_STR_TO_INT[arg_type]
                                    if act_type not in ALFRED_ACTION_LIST:
                                        continue
                                    is_end = False
                                    break
                                except:
                                    continue
                            if is_end:
                                print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
                                self.counter += 1
                                return rollout

                            hl_action = AlfredSubgoal.from_type_str_and_arg_id(act_type, arg_id)
                            hl_action = hl_action.to('cuda')

                            action = self.agent.act(observation, hl_action)

                            ll_action_list = []
                            # ll_action_list.append(action.action_type)

                            fail_count = 0

                        else:
                            ll_action_list.append(action.action_type)


            print(f"Finished rollout: {self.counter}, length: {len(rollout)}")
            self.counter += 1
            return rollout

    def _parse_policy_from_string(self, policy_text: str) -> list:
        """Helper function to parse a policy from a string returned by the LLM."""
        policy_list = []
        after_revise = False
        lines = policy_text.split('\n')
        for line in lines:
            if 'revise' in line.lower():
                after_revise = True
            if not after_revise:
                continue

            policy_dict = {}
            if '(' in line and ')' in line:
                idx_1 = line.find('(')
                idx_2 = line.find(')')
                line_content = line[idx_1 + 1: idx_2]
                split_arguments = line_content.split(',')
                if len(split_arguments) == 2:
                    policy_dict['action'] = split_arguments[0].strip()
                    policy_dict['object'] = split_arguments[1].strip()
                elif len(split_arguments) == 1:
                    policy_dict['action'] = split_arguments[0].strip()
                    policy_dict['object'] = ''
                else:
                    policy_dict['action'] = split_arguments[0].strip()
                    # Handle cases like "put(object, in, receptacle)"
                    if 'put' in split_arguments[0].strip().lower():
                        policy_dict['object'] = split_arguments[2].strip()
                    else:
                        policy_dict['object'] = split_arguments[1].strip()
                policy_list.append(policy_dict)
        return policy_list

    def replan_policies(self, raw_image, task_description, qa_file, policy, action, argument, seen_items, idx):
        self.logger.info(f"Replanning for task: {task_description}")

        qa_file = open(qa_file, 'r')
        subtask = ""
        stop = True
        while True:
            line = qa_file.readline()
            if len(line) == 1:
                continue
            elif not line:
                break

            if not stop:
                if 'Answer' in line:
                    subtask = line.split(':')[1]
                    break

            if task_description in line or line in task_description:
                stop = False

        policy_str = ""
        failed_index = 0
        for i, p in enumerate(policy):
            policy_str += str(i + 1) + ". " + "(" + p["action"] + ", " + p["object"] + ")\n"
        for i, p in enumerate(policy):
            if p["action"] == action and p["object"] == argument and i >= idx - 1:
                failed_index = i + 1
                break
        failed_action = str(failed_index) + ". " + "(" + action + ", " + argument + ")"
        Replanning_QA_GPT = Replanning_QA(task_description)

        image = Replanning_QA_GPT.image(raw_image)
        res, action, ans = Replanning_QA_GPT.blip(failed_action, image)

        if "no" in ans:
            cause = Replanning_QA_GPT.cause(image,res,action)
            try:
                new_policy_text = Replanning_QA_GPT.Replanning_prompt(policy_str, failed_action, str(seen_items), res, cause)
            except:
                self.logger.error("GPT is not responding during replanning.")
                return [] # Return empty policy on failure

            # Log the details for debugging instead of writing to a passed file handle
            self.logger.info("--- Replanning Details ---")
            self.logger.info(f"Task: {task_description}")
            self.logger.info(f"Original Policy: {policy_str}")
        
        else:
            print('No replanning. Back to HLSM.')
            policy_list = policy
    
        return policy_list

    def rollout_and_send(self):
        self._load_agent_state_from_ps()
        rollout = self.rollout()

        # Send to the dataset process
        self.dataset_process.add_rollout.remote(rollout)

        # Write metrics to tensorboard
        #if self.writer is not None:
        #    metrics = get_multiple_rollout_metrics_bw([rollout])
        #    self.writer.add_scalar_dict("tsa_rollout", metrics)
        #    self.writer.inc_iter()
        return


@ray.remote(num_cpus=1, num_gpus=0)
class RolloutActor(RolloutActorLocal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)