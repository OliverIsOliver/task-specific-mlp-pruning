import gc
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


MODEL = 'google/gemma-2-2b'
REVISION = 'c5ebcd40d208330abc697524c919956e692655cf'
CACHE = Path(os.environ.get('PRUNING_MODEL_CACHE', Path.home() / '.cache' / 'huggingface' / 'hub'))


def load_model(source,device):
    model = AutoModelForCausalLM.from_pretrained(source,dtype=torch.bfloat16,
        attn_implementation='eager').to(device).eval()
    # Mask training changes gates, not the pretrained weights.
    model.requires_grad_(False)
    return model


def prune_mlp(model, removed, device):
    for i, block in enumerate(model.model.layers):
        index = torch.tensor(np.flatnonzero(~removed[i]), device=device)
        # Delete gate/up rows and matching down columns to make genuinely smaller matrices.
        for name, axis in [('gate_proj',0), ('up_proj',0), ('down_proj',1)]:
            layer = getattr(block.mlp, name)
            layer.weight = torch.nn.Parameter(layer.weight.detach().index_select(axis,index), requires_grad=False)
            layer.out_features, layer.in_features = layer.weight.shape
    assert removed.size-sum(b.mlp.down_proj.in_features for b in model.model.layers)==int(removed.sum())


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
