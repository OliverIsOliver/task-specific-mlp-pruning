import numpy as np
import torch


def fisher_scores(loss, activations, retain_graph=False):
    # Estimate sensitivity to a multiplicative gate on each MLP neuron.
    gradients = torch.autograd.grad(loss, activations, retain_graph=retain_graph)
    g = torch.stack([(a.detach().float() * grad.float()).sum((0, 1))
                     for a, grad in zip(activations, gradients)]).cpu().numpy()
    # Sum token contributions before squaring; average these scores over training examples.
    return (.5 * g.astype(np.float64)**2).astype(np.float32)


def score(row, tokenizer, model, device):
    prefix = tokenizer(row['prompt']).input_ids
    answer = tokenizer(' '+row['answer'], add_special_tokens=False).input_ids
    activations = [None]*len(model.model.layers)

    def capture(i):
        def hook(module, args):
            activations[i] = args[0]
        return hook

    # Capture MLP activations just before they are projected back into the hidden state.
    hooks = [block.mlp.down_proj.register_forward_pre_hook(capture(i))
             for i, block in enumerate(model.model.layers)]
    try:
        # Teacher forcing: measure loss only on the known training answer tokens.
        logits = model(input_ids=torch.tensor([prefix+answer[:-1]], device=device),
                       use_cache=False, logits_to_keep=len(answer)).logits
        loss = torch.nn.functional.cross_entropy(logits[0].float(), torch.tensor(answer, device=device))
        scores = fisher_scores(loss, activations)
        return scores, dict(loss=float(loss.detach()), prompt_tokens=len(prefix), answer_tokens=len(answer))
    finally:
        for hook in hooks:
            hook.remove()


def global_mask(scores, fraction=.2):
    # Rank across all layers together, so each layer can lose a different number.
    removed = np.argsort(scores.ravel(), kind='stable')[:int(scores.size*fraction)]
    result = np.zeros(scores.size, dtype=bool)
    result[removed] = True
    return result.reshape(scores.shape)
