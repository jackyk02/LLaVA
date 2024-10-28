# run_reward_model.py
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.language_model.llava_reward import LlavaLlamaForReward
import torch
from llava.utils import disable_torch_init
import requests

import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def convert_to_reward_model(model):
    """Convert LLaVA model to reward model"""
    config = model.config
    reward_model = LlavaLlamaForReward(config)
    
    # Copy weights from pretrained model except lm_head
    pretrained_dict = model.state_dict()
    model_dict = reward_model.state_dict()
    
    # Filter out lm_head and unmatched keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and not k.startswith('lm_head')}
    
    # Update model dict and load weights
    model_dict.update(pretrained_dict)
    reward_model.load_state_dict(model_dict, strict=False)
    
    # Copy vision tower if it exists
    if hasattr(model.model, 'vision_tower'):
        reward_model.model.vision_tower = model.model.vision_tower
        
    return reward_model
    
def evaluate_reward(
    model_path: str,
    image_file: str,
    prompt: str,
    device: str = "cuda"
) -> float:
    """
    Evaluate the reward for a given image and prompt
    """
    # Disable torch init
    disable_torch_init()
    
    # Load the model
    model_name = get_model_name_from_path(model_path)
    tokenizer, base_model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name
    )
    
    # Convert to reward model
    model = convert_to_reward_model(base_model)
    model = model.to(device)
    model.eval()
    
    # Prepare the image
    image = load_image(image_file)
    image_tensor = process_images([image], image_processor, model.config).to(device)
    image_sizes = [image.size]
    
    # Prepare the prompt
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"
        
    conv = conv_templates[conv_mode].copy()
    
    # Add image tokens
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    prompt = image_token_se + "\n" + prompt
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Get reward prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            return_dict=True  # Explicitly set return_dict to True
        )
        
        # Handle both tuple and dict returns
        if isinstance(outputs, tuple):
            reward = outputs[0].item()  # First element is reward
        else:
            reward = outputs.reward.item()
    
    return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    reward = evaluate_reward(
        model_path=args.model_path,
        image_file=args.image_file,
        prompt=args.prompt,
        device=args.device
    )
    
    print(f"Reward value: {reward}")