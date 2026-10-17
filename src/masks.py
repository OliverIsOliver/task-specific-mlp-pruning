import hashlib
import json
import time

import numpy as np
import torch

from .storage import write_json


def digest(mask):
    return hashlib.sha256(mask.tobytes()).hexdigest()


def answer_loss(row, tokenizer, model, device):
    prefix = tokenizer(row['prompt']).input_ids
    answer = tokenizer(' ' + row['answer'], add_special_tokens=False).input_ids
    ids = torch.tensor([prefix + answer[:-1]], device=device)
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(answer)).logits
    return torch.nn.functional.cross_entropy(logits[0].float(), torch.tensor(answer, device=device))


@torch.no_grad()
def mean_loss(rows, tokenizer, model, device):
    return float(np.mean([float(answer_loss(r, tokenizer, model, device)) for r in rows]))


class LearnedMask:
    def __init__(self, model, fisher, fraction=.2):
        self.model = model
        self.count = int(fisher.size * fraction)
        # Initialize gate scores in Fisher order; low-importance neurons start lowest.
        ranks = np.empty(fisher.size)
        ranks[np.argsort(fisher.ravel(), kind='stable')] = np.arange(fisher.size)
        p = .05 + .9 * ranks.reshape(fisher.shape) / (fisher.size - 1)
        device = model.model.layers[0].mlp.down_proj.weight.device
        self.logits = torch.nn.Parameter(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32, device=device))
        self.scales = None
        self.hooks = [block.mlp.down_proj.register_forward_pre_hook(self.hook(i))
                      for i, block in enumerate(model.model.layers)]

    def hook(self, i):
        def apply(module, args):
            return (args[0] * self.scales[i].to(args[0].dtype),)
        return apply

    def project(self):
        values = self.logits.detach().cpu().numpy()
        removed = np.zeros(values.size, dtype=bool)
        # Keep the removal count fixed while allowing the selected neurons to change.
        removed[np.argsort(values.ravel(), kind='stable')[:self.count]] = True
        return removed.reshape(values.shape)

    def apply(self, removed, training=False):
        hard = torch.tensor(~removed, dtype=torch.float32, device=self.logits.device)
        p = self.logits.sigmoid()
        # Forward uses binary gates; backward uses sigmoid gradients (straight-through).
        self.scales = hard + (p - p.detach()) if training else hard

    def close(self):
        for hook in self.hooks:
            hook.remove()


def learn(rows, tokenizer, model, device, fisher, reference, root, steps, batch_size, lr, fraction=.2,
          example_weights=None):
    versions = [p._version for p in model.parameters()]
    assert not any(p.requires_grad for p in model.parameters())
    gates = LearnedMask(model, fisher, fraction)
    initial = removed = gates.project()
    assert np.array_equal(initial, reference)
    # Train only gate logits. Original model weights remain frozen.
    optimizer = torch.optim.Adam([gates.logits], lr=lr)
    # The saved runs use 25 batches of 2: one shuffled pass over 50 training examples.
    order = np.random.default_rng(821).permutation(len(rows))
    weights = {r['id']:float(example_weights[r['id']]) for r in rows} if example_weights else None
    normalizer = float(np.mean(list(weights.values()))) if weights else 1.
    trace = []
    began = time.perf_counter()
    try:
        gates.apply(removed)
        initial_loss = mean_loss(rows, tokenizer, model, device)
        print(f'Initial Fisher-mask calibration loss: {initial_loss:.6f}', flush=True)
        for step in range(steps):
            batch = [rows[order[(step * batch_size + j) % len(rows)]] for j in range(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            weighted_losses = []
            for row in batch:
                gates.apply(removed, training=True)
                loss = answer_loss(row, tokenizer, model, device)
                losses.append(float(loss.detach()))
                weight = weights[row['id']]/normalizer if weights else 1.
                weighted_losses.append(float(loss.detach())*weight)
                (loss * weight / len(batch)).backward()
            gradient = gates.logits.grad.detach()
            assert torch.isfinite(gradient).all()
            muted = torch.tensor(removed, device=device)
            muted_gradients = int(torch.count_nonzero(gradient[muted]))
            # Update gate scores, then select a new mask with the same neuron budget.
            optimizer.step()
            removed = gates.project()
            assert int(removed.sum()) == int(reference.sum())
            trace.append(dict(step=step+1, training_ids=[r['id'] for r in batch],
                batch_loss_before_update=float(np.mean(losses)), removed=int(removed.sum()),
                restored_from_initial=int((initial & ~removed).sum()),
                muted_neurons_with_nonzero_gradient=muted_gradients, mask_sha256=digest(removed)))
            if weights:
                trace[-1].update(example_weights=[weights[r['id']] for r in batch],
                    weighted_batch_loss_before_update=float(np.mean(weighted_losses)))
            write_json(root / 'trace.json', trace)
            if (step+1) % 5 == 0:
                print(f'Joint update {step+1}/{steps}: restored {trace[-1]["restored_from_initial"]} '
                      'initially cut neurons', flush=True)
        gates.apply(removed)
        final_loss = mean_loss(rows, tokenizer, model, device)
        assert versions == [p._version for p in model.parameters()]
        assert all(p.grad is None for p in model.parameters())
        # Save the final mask, its Fisher initialization, and the learned gate scores.
        np.savez_compressed(root / 'mask.npz', removed=removed, initial=initial,
                            logits=gates.logits.detach().cpu().numpy())
        stats = dict(initial_calibration_loss=initial_loss, final_calibration_loss=final_loss,
            steps=steps, backward_passes=steps*batch_size, seconds=time.perf_counter()-began,
            selected='Final fixed-budget iterate; no checkpoint selection using test results.',
            original_parameter_versions_unchanged=True, original_parameter_gradients_absent=True,
            mask_sha256=digest(removed), restored_from_initial=int((initial & ~removed).sum()))
        if weights:
            stats.update(example_weights=weights, weight_normalizer=normalizer,
                objective='Mean of example loss times fixed weight divided by training-wide mean weight.')
        write_json(root / 'selection.json', stats)
        print(json.dumps(stats, indent=2), flush=True)
        return removed
    finally:
        gates.close()
