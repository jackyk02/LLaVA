from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import ModelOutput
from llava.model.language_model.llava_llama import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.multimodal_encoder.builder import build_vision_tower
import json
from PIL import Image
import requests
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class LlavaRewardModelOutput(ModelOutput):
    """
    Output type of LlavaRewardModel.
    
    Args:
        reward (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward predictions for each input.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of hidden states from the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of attention weights from the model.
    """
    
    reward: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class LlavaRewardConfig(LlamaConfig):
    model_type = "llava_reward"

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaRewardConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaForReward(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaRewardConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        
        # Remove the language modeling head
        if hasattr(self, 'lm_head'):
            del self.lm_head
            
        # Add reward head
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1)
        )

        # Initialize vision encoder
        vision_tower = build_vision_tower(config)
        if vision_tower is not None:
            self.model.vision_tower = vision_tower
        self.model.config.use_cache = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[List[List[int]]] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, LlavaRewardModelOutput]:
        # Set return_dict to True if not specified
        if return_dict is None:
            return_dict = True

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state if return_dict else outputs[0]
        
        # Pool the sequence
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            last_hidden_state = last_hidden_state * attention_mask
            pooled = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)
        else:
            pooled = last_hidden_state.mean(dim=1)

        # Calculate reward value
        reward = self.reward_head(pooled)

        if not return_dict:
            outputs = (reward,) + outputs[1:]
            return outputs

        return LlavaRewardModelOutput(
            reward=reward,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                    inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


# Register the model and config
AutoConfig.register("llava_reward", LlavaRewardConfig)
AutoModelForCausalLM.register(LlavaRewardConfig, LlavaLlamaForReward)