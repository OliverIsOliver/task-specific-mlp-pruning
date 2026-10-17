import json
from pathlib import Path

import numpy as np


SITE = Path(__file__).resolve().parent
ROOT = SITE.parent


def read(path):
    return json.loads(path.read_text())


def export_task(title, tab, item, task):
    data = ROOT / 'data' / task
    source = ROOT / 'masks' / task
    with np.load(source / 'fisher.npz') as saved:
        scores = saved['fisher']
    percentile = np.empty(scores.size)
    percentile[np.argsort(scores.ravel(), kind='stable')] = np.linspace(0, 1, scores.size)
    percentile = percentile.reshape(scores.shape)
    # Fisher determines membership; average neuron ID determines display order.
    members = [sorted(np.array_split(np.argsort(-layer, kind='stable'), 10),
                      key=lambda ids: ids.mean()) for layer in scores]
    groups = [[{
        'size': len(ids), 'meanNeuronId': round(float(ids.mean()), 2),
        'percentile': round(float(percentile[layer, ids].mean()) * 100, 2),
    } for ids in layer_groups] for layer, layer_groups in enumerate(members)]
    prompts = read(data / 'test-prompts.json')
    targets = read(data / 'test-targets.json')
    training = read(data / 'train.json')
    slug = tab.lower().replace(' ', '-')
    dataset_dir = SITE / 'public/datasets' / slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    method = 'Fisher + mask optimization'
    for split, examples in [('training', training),
                            ('testing', [dict(row, answer=targets[row['id']]) for row in prompts])]:
        payload = dict(task=title, model='Gemma 2 2B', method=method, source=f'data/{task}',
                       split=split, examples=examples)
        (dataset_dir / f'{split}.json').write_text(json.dumps(payload, indent=2) + '\n')
    modes, outputs = [], {}
    for cut in (0, 10, 20, 30):
        key = f'pruned_{cut}' if cut else 'untouched'
        if cut:
            with np.load(source / f'cut-{cut}.npz') as saved:
                mask = saved['removed']
        else:
            mask = np.zeros_like(scores, dtype=bool)
        assert int(mask.sum()) == int(scores.size * cut / 100)
        rows = read(ROOT / 'results' / task / f'cut-{cut}/outputs.json')
        outputs[key] = {row['id']: row for row in rows}
        assert len(rows) == len(outputs[key]) == len(prompts) == 100
        modes.append(dict(
            key=key, cut=cut, label=f'{100-cut}% model', removed=int(mask.sum()),
            correct=sum(row['exact_match'] for row in rows),
            remaining=[[int((~mask[layer, ids]).sum()) for ids in layer_groups]
                       for layer, layer_groups in enumerate(members)],
            opacity=[[round(float((0.3 + 0.7 * percentile[layer, ids].mean()) *
                                  (~mask[layer, ids]).mean()), 6) for ids in layer_groups]
                     for layer, layer_groups in enumerate(members)],
        ))
    samples = [dict(
        id=row['id'], transcript=row['transcript'], expected=json.loads(targets[row['id']]),
        answers={key: dict(text=rows[row['id']]['response'],
                           correct=rows[row['id']]['exact_match'],
                           truncated=rows[row['id']]['truncated']) for key, rows in outputs.items()},
    ) for row in prompts]
    return dict(
        title=title, tab=tab, item=item, model='Gemma 2 2B', method=method,
        scope='Integer addition' if task == 'addition' else '', optimized=True,
        training=len(training), maxTokens=128 if item == 'Order' else 64,
        datasets={split: f'datasets/{slug}/{split}.json' for split in ('training', 'testing')},
        layers=scores.shape[0], total=scores.size, source=f'masks/{task}',
        groups=groups, modes=modes, samples=sorted(samples, key=lambda row: row['id']),
    )


def main():
    tasks = [
        export_task('Food orders', 'Food orders', 'Order', 'food'),
        export_task('Scheduling appointments', 'Appointments', 'Appointment', 'appointments'),
        export_task('Understanding policies', 'Policies', 'Case', 'returns'),
        export_task('Calculating arithmetic', 'Arithmetic', 'Question', 'addition'),
    ]
    (SITE / 'src/data.json').write_text(json.dumps(tasks, separators=(',', ':')) + '\n')
    for task in tasks:
        print(f"{task['title']}: {len(task['samples'])} saved cases; "
              f"correct = {[mode['correct'] for mode in task['modes']]}")


if __name__ == '__main__':
    main()
