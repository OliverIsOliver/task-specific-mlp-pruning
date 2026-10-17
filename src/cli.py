import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved masks or fit new Fisher-initialized joint masks.')
    parser.add_argument('--tasks', nargs='+', choices=['food', 'appointments', 'returns', 'addition'],
                        default=['food', 'appointments', 'returns', 'addition'])
    parser.add_argument('--cuts', nargs='+', type=int, choices=[0, 10, 20, 30], default=[0, 10, 20, 30])
    parser.add_argument('--fit', action='store_true', help='Recompute training Fisher scores and learn new masks.')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from .evaluation import correct
    from .fisher import global_mask, score
    from .generation import generate_addition, generate_food, generate_json
    from .masks import learn
    from .model import CACHE, MODEL, REVISION, load_model, prune_mlp, release_memory
    from .storage import read_json, write_json

    source = snapshot_download(MODEL, revision=REVISION, cache_dir=CACHE)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    for task in args.tasks:
        folder = args.output / task
        folder.mkdir(parents=True, exist_ok=True)
        data = ROOT / 'data' / task
        train, prompts, targets = [read_json(data / name) for name in ('train.json', 'test-prompts.json', 'test-targets.json')]
        # Only training examples contribute to Fisher scores and mask learning.
        if args.fit:
            model = load_model(source, device)
            model.enable_input_require_grads()
            try:
                fisher = np.mean([score(row, tokenizer, model, device)[0].astype(np.float64)
                                  for row in train], axis=0)
            finally:
                model.disable_input_require_grads()
            np.savez_compressed(folder / 'fisher.npz', fisher=fisher)
            del model
            release_memory()
        # Start each cut from fresh original weights; never prune an already-pruned model.
        for cut in args.cuts:
            dest = folder / f'cut-{cut}'
            dest.mkdir(exist_ok=True)
            model = load_model(source, device)
            if cut:
                if args.fit:
                    removed = learn(train, tokenizer, model, device, fisher, global_mask(fisher, cut / 100),
                                    dest, 25, 2, .01, cut / 100)
                else:
                    removed = np.load(ROOT / 'masks' / task / f'cut-{cut}.npz')['removed']
                prune_mlp(model, removed, device)
            if task == 'food':
                rows = [generate_food(row, tokenizer, model, device, grade=False, max_tokens=128) for row in prompts]
            elif task == 'addition':
                rows = generate_addition(prompts, tokenizer, model, device, dest / 'generated.json')
            else:
                rows = generate_json(prompts, tokenizer, model, device, dest / 'generated.json')
            # Test answers are consulted only after generation, for scoring.
            outputs = []
            for row in rows:
                expected = targets[row['id']]
                matched = correct(task, expected, row['response'])
                outputs.append(dict(row, expected=expected, exact_match=matched))
            result = dict(correct=sum(row['exact_match'] for row in outputs), total=len(outputs))
            write_json(dest / 'outputs.json', outputs)
            write_json(dest / 'results.json', result)
            print(task, cut, result, flush=True)
            del model
            release_memory()


if __name__ == '__main__':
    main()
