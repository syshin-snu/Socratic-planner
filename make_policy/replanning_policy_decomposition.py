import os
import re
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import numpy as np
from typing import Union, Tuple

import openai

# Use a relative import for better package portability.
from prompt.prompt_replanning_blip import ReplanningPrompt


def setup_openai_api():
    """Loads the OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key

class Replanner:
    """
    A class to handle policy replanning using visual information from BLIP-2
    and reasoning from an LLM.
    """
    def __init__(self, device, llm_model="gpt-3.5-turbo", max_tokens=500):
        self.device = device
        self.llm_model = llm_model
        self.max_tokens = max_tokens

        # Load BLIP-2 model
        print(f"Loading BLIP-2 model to {self.device}...")
        self.blip_model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=self.device
        )
        print("BLIP-2 model loaded.")
    
    def _process_image(self, image_source: Union[str, np.ndarray]) -> torch.Tensor:
        """Loads and preprocesses an image from a path or a numpy array."""
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image file not found at: {image_source}")
            raw_image = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, np.ndarray):
            raw_image = Image.fromarray(image_source).convert("RGB")
        else:
            raise TypeError(f"Unsupported image source type: {type(image_source)}")

        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        return image
    
    def _get_visual_context(self, failed_action: str, image: torch.Tensor) -> Tuple[str, str, str]:
        """Uses BLIP-2 to get context from the image about the failed action."""
        # Describe the scene
        scene_desc = self.blip_model.generate({
            "image": image,
            "prompt": "This is a scene of"
        })[0]

        # Use regex to robustly parse the failed action string, e.g., "2. (Action, Arg1, Arg2)"
        match = re.search(r'\(\s*(\w+)\s*,\s*([^,)]+)(?:\s*,\s*([^)]+))?\s*\)', failed_action)
        if not match:
            # Fallback for simple action phrases if regex fails
            action_phrase = failed_action
        else:
            action, arg1, arg2 = match.groups()
            action_verb = action.replace('Object', '').lower()
            
            if 'put' in action_verb and arg2:
                action_phrase = f"put the {arg1.strip().lower()} on the {arg2.strip().lower()}"
            else:
                action_phrase = f"{action_verb} the {arg1.strip().lower()}"


        # Ask if the action is possible
        is_possible_answer = self.blip_model.generate({
            "image": image,
            "prompt": f"This is a scene of {scene_desc}. Is the agent able to {action_phrase}?"
        })[0]
        return scene_desc, action_phrase, is_possible_answer
    
    def _get_failure_cause(self, image: torch.Tensor, scene_desc: str, action_phrase: str) -> str:
        """Asks BLIP-2 for the cause of the failure."""
        cause = self.blip_model.generate({
            "image": image,
            "prompt": f"This is a scene of {scene_desc}. The agent is not able to {action_phrase} because"
        })
        return cause[0]
        
    def _call_llm(self, messages: list) -> str:
        """Calls the OpenAI ChatCompletion API."""
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
        )
        return response["choices"][0]["message"]["content"]
        
    def generate_replanned_policy(
        self, 
        instruction: str, 
        image_source: Union[str, np.ndarray], 
        completed_plan: str, 
        failed_action: str, 
        visible_objects: list
    ) -> str:
        """The main method to generate a replanned policy."""
        image = self._process_image(image_source)
        scene_desc, action_phrase, is_possible_answer = self._get_visual_context(failed_action, image)

        if 'no' in is_possible_answer.lower():
            cause = self._get_failure_cause(image, scene_desc, action_phrase)
            user_prompt = ReplanningPrompt(instruction, completed_plan, failed_action, visible_objects, scene_desc, cause)
            messages = [{"role": "user", "content": user_prompt}]
            
            print("\n--- Replanning needed. Asking LLM for a new plan. ---")
            replanned_policy = self._call_llm(messages)
            return replanned_policy
        else:
            print("\n--- No replanning needed. Visual model indicates action is possible. ---")
            return "No replanning needed. The original plan should proceed."
