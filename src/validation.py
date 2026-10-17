import json
from pathlib import Path

import numpy as np

from .evaluation import correct

ROOT = Path(__file__).resolve().parents[1]


# Check saved datasets, masks, and grading without loading Gemma.
def main():
    read = lambda p: json.loads(p.read_text())
    summary = read(ROOT / 'results/summary.json')
    for task in summary:
        data = ROOT / 'data' / task
        train, prompts, targets = [read(data / name) for name in ('train.json', 'test-prompts.json', 'test-targets.json')]
        assert len(train) == 50 and len(prompts) == len(targets) == 100
        assert not ({row['prompt'] for row in train} & {row['prompt'] for row in prompts})
        ids = {row['id'] for row in prompts}
        assert len(ids) == 100 and ids == set(targets)
        fisher = np.load(ROOT / 'masks' / task / 'fisher.npz')['fisher']
        assert fisher.shape == (26, 9216) and np.isfinite(fisher).all()
        for cut in (0, 10, 20, 30):
            dest = ROOT / 'results' / task / f'cut-{cut}'
            rows = read(dest / 'outputs.json')
            assert len(rows) == 100 and {row['id'] for row in rows} == ids
            for row in rows:
                expected = targets[row['id']]
                matched = correct(task, expected, row['response'])
                assert matched == row['exact_match']
            assert sum(row['exact_match'] for row in rows) == summary[task][str(cut)]['correct']
            if cut:
                with np.load(ROOT / 'masks' / task / f'cut-{cut}.npz') as saved:
                    removed, logits = saved['removed'], saved['logits']
                assert removed.shape == fisher.shape and removed.dtype == bool
                assert int(removed.sum()) == int(fisher.size * cut / 100)
                projected = np.zeros(removed.size, dtype=bool)
                projected[np.argsort(logits.ravel(), kind='stable')[:int(removed.sum())]] = True
                assert np.array_equal(projected.reshape(removed.shape), removed)
        print(task, 'verified: 50 training, 100 test, 4 model sizes')
    print('Verified 12 masks and independently graded all 1,600 saved responses. No model loaded.')


if __name__ == '__main__':
    main()
