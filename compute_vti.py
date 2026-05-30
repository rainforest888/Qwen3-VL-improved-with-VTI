import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any
from tqdm import tqdm
from collections import Counter

def extract_visual_hidden_states(
    model,
    processor,
    images: List,
    mask_ratio: float = 0.3,
    num_trials: int = 3
) -> torch.Tensor:

    delta_list = []

    for img in tqdm(images, desc="Extracting visual states"):

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 原图 forward
        with torch.no_grad():
            outputs_orig = model(
                **inputs,
                output_hidden_states=True
            )

        orig_h = outputs_orig.hidden_states[-1][:, -1, :]
        orig_h = orig_h.detach().cpu().float()

        # 多次mask
        for _ in range(num_trials):

            masked_inputs = {}

            for k, v in inputs.items():

                if k == "pixel_values":

                    mask = (
                        torch.rand_like(v) >= mask_ratio
                    ).to(v.dtype)

                    masked_inputs[k] = v * mask

                else:
                    masked_inputs[k] = v

            with torch.no_grad():

                outputs_mask = model(
                    **masked_inputs,
                    output_hidden_states=True
                )

            mask_h = outputs_mask.hidden_states[-1][:, -1, :]
            mask_h = mask_h.detach().cpu().float()

            delta = orig_h - mask_h

            delta_list.append(delta.squeeze(0))

    return torch.stack(delta_list, dim=0)

def extract_textual_hidden_states(model, processor, demo_data: List[Dict[str, Any]]) -> torch.Tensor:
    delta_list = []
    for item in tqdm(demo_data, desc="Extracting textual states"):
        img = item['image']
        good = item['good_text']
        bad = item['bad_text']
        inputs_good = processor(text=good, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs_good = model(**inputs_good.to(model.device), output_hidden_states=True)
            last_h_good = outputs_good.hidden_states[-1][:, -1, :].cpu().float()
        inputs_bad = processor(text=bad, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs_bad = model(**inputs_bad.to(model.device), output_hidden_states=True)
            last_h_bad = outputs_bad.hidden_states[-1][:, -1, :].cpu().float()
        delta = last_h_good - last_h_bad
        delta_list.append(delta)
    return torch.cat(delta_list, dim=0)

def compute_visual_direction(model, processor, demo_images: List, rank: int = 1, **kwargs) -> torch.Tensor:
    deltas = extract_visual_hidden_states(model, processor, demo_images, **kwargs)
    pca = PCA(n_components=rank)
    pca.fit(deltas.numpy())
    direction = torch.tensor(pca.components_[0], dtype=torch.float32)
    return direction / (torch.norm(direction) + 1e-8)

def compute_textual_direction(model, processor, demo_data: List[Dict[str, Any]], rank: int = 1) -> torch.Tensor:
    deltas = extract_textual_hidden_states(model, processor, demo_data)
    pca = PCA(n_components=rank)
    pca.fit(deltas.numpy())
    direction = torch.tensor(pca.components_[0], dtype=torch.float32)
    return direction / (torch.norm(direction) + 1e-8)