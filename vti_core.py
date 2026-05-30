import torch
import torch.nn as nn
from typing import List, Optional, Union

class VTILayer(nn.Module):
    def __init__(self, direction: torch.Tensor, alpha: float = 0.9, normalize: bool = True):
        super().__init__()
        self.register_buffer("direction", direction.clone().detach().to(dtype=torch.bfloat16, device="cuda"))
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.direction.shape[-1]:
            return hidden_states
        dir_expanded = self.direction.view(1, 1, -1).expand_as(hidden_states)
        output = hidden_states + self.alpha * dir_expanded
        if self.normalize:
            orig_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
            cur_norm = torch.norm(output, dim=-1, keepdim=True)
            output = output * (orig_norm / (cur_norm + 1e-8))
        return output

class _VTIMLPWrapper(nn.Module):
    def __init__(self, mlp: nn.Module, direction: torch.Tensor, alpha: float, normalize: bool):
        super().__init__()
        self.original_mlp = mlp
        self.vti = VTILayer(direction, alpha, normalize)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vti(self.original_mlp(x))

def get_layers(model: nn.Module, target_keywords: Optional[List[str]] = None) -> nn.ModuleList:
    target_keywords = target_keywords or ['model.layers', 'transformer.h', 'layers', 'blocks', 'encoder.layers']
    def _find(obj, depth=0):
        if depth > 5: return None
        for name, child in obj.named_children():
            if isinstance(child, nn.ModuleList) and any(kw in name.lower() for kw in ['layer', 'block']):
                return child
            result = _find(child, depth+1)
            if result is not None: return result
        return None
    layers = _find(model)
    if layers is None: raise ValueError("未找到模型层")
    return layers

def add_vti_layers(
    model: nn.Module, directions: Union[torch.Tensor, List[torch.Tensor]],
    alpha: float = 0.9, layer_indices: Optional[List[int]] = None,
    normalize: bool = True, target_keywords: Optional[List[str]] = None
) -> nn.Module:
    layers = get_layers(model, target_keywords)
    num_layers = len(layers)
    if isinstance(directions, torch.Tensor):
        if directions.dim() == 1: dir_list = [directions.clone() for _ in range(num_layers)]
        elif directions.dim() == 2: dir_list = [directions[i].clone() for i in range(num_layers)]
    else:
        dir_list = directions
    if len(dir_list) == 1: dir_list = dir_list * num_layers

    if layer_indices is None: layer_indices = list(range(num_layers))

    for i in layer_indices:
        if i >= num_layers: continue
        original_layer = layers[i]
        mlp_module, target_attr = None, None
        for attr in ['mlp', 'feed_forward', 'ffn', 'w2']:
            if hasattr(original_layer, attr):
                mlp_module = getattr(original_layer, attr)
                target_attr = attr
                break
        if mlp_module is None: continue
        wrapped_mlp = _VTIMLPWrapper(mlp_module, dir_list[i], alpha, normalize)
        setattr(original_layer, target_attr, wrapped_mlp) # 精准注入
    return model

def remove_vti_layers(model: nn.Module, target_keywords: Optional[List[str]] = None) -> nn.Module:
    layers = get_layers(model, target_keywords)
    for layer in layers:
        for attr in ['mlp', 'feed_forward', 'ffn', 'w2']:
            if hasattr(layer, attr) and isinstance(getattr(layer, attr), _VTIMLPWrapper):
                setattr(layer, attr, getattr(layer, attr).original_mlp)
    return model
