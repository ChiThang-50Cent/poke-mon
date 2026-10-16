"""Attach the single W4.4 20-battle sanity check, not a policy benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np

from pokeai.data.w3 import load_split
from pokeai.training.value_metrics import value_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in
               (args.run / 'battles.jsonl').read_text().splitlines()]
    summary = json.loads((args.run / 'summary.json').read_text())
    probabilities, outcomes = [], []
    for row in records:
        if row['error'] is not None:
            raise RuntimeError(f"Smoke battle failed: {row['error']}")
        if row['winner'] not in {'player', 'opponent'}:
            raise RuntimeError('Smoke contains unresolved or tied outcome')
        values = row['values']['player']
        probabilities.extend(values)
        outcomes.extend([int(row['winner'] == 'player')] * len(values))
    if len(records) != 20:
        raise RuntimeError(f'Expected 20 battles, got {len(records)}')
    illegal = summary['neural']['player']['illegal_actions']
    if illegal:
        raise RuntimeError(f'Smoke contains {illegal} illegal actions')
    metrics = value_metrics(np.asarray(probabilities), np.asarray(outcomes))
    smoke = {'run_dir': str(args.run), 'battles': len(records),
             'errors': 0, 'illegal_actions': illegal, 'value': metrics,
             'summary': summary,
             'interpretation': 'Live Brier on 20 battles only; sanity check, '
                               'not a policy-quality or win-rate claim'}
    report_path = Path('reports/w4_value_metrics.json')
    report = json.loads(report_path.read_text())
    report['live_smoke'] = smoke
    val = load_split('val', 'datasets/w4')
    final = val['numbers'][val['offsets'][1:] - 1]
    own_remaining = (final[:, 1] > 0).astype(int)
    own_remaining += (final[:, 28:33] > 0).sum(axis=1)
    opponent_remaining = np.rint(final[:, 0] * 6).astype(int)
    trivially_terminal = (own_remaining == 0) | (opponent_remaining == 0)
    report['leakage_audit']['final_observation_terminal_hp_check'] = {
        'trajectories': len(final),
        'zero_own_or_opponent_remaining': int(trivially_terminal.sum()),
        'fraction': float(trivially_terminal.mean()),
        'method': 'Own active HP + positive available-switch HP; opponents_remaining',
        'interpretation': 'Only checks explicit terminal HP counts, not tactical '
                          'inevitability. See final-step model accuracy separately.',
    }
    val_base = report['references']['constant_0.5']['first_step']['base_rate']
    report['leakage_audit']['turn0_near_base_rate'] = {
        row['mode']: {
            'mean_probability': row['value']['first_step']['mean_probability'],
            'trajectory_base_rate': val_base,
            'absolute_difference': abs(
                row['value']['first_step']['mean_probability'] - val_base
            ),
            'within_0.05': abs(
                row['value']['first_step']['mean_probability'] - val_base
            ) <= 0.05,
        }
        for row in report['runs']
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(smoke, indent=2))


if __name__ == '__main__':
    main()
