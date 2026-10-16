"""Fixed W4.4 probe/joint ablation; W4 validation only, never action selection.

Run under the AGENTS.md systemd scope. Both runs execute sequentially once.
"""

import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pokeai.data.w3 import load_split
from pokeai.data.w4 import load_train_tier
from pokeai.data.w4_outcomes import build_outcomes, load_outcomes
from pokeai.models.pokedecision import load_pokedecision
from pokeai.training.train_bc import (
    TrajectorySequences,
    pad_sequences,
    run_epoch,
    train_arch,
)
from pokeai.training.value_metrics import rating_reference, trajectory_metrics


def probabilities(model, arrays, device):
    loader = DataLoader(
        TrajectorySequences(arrays), batch_size=16, collate_fn=pad_sequences
    )
    result = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            _, logits, _ = model.forward_with_value(
                batch['tokens'].long().to(device), batch['numbers'].to(device)
            )
            result.append(logits.sigmoid().cpu().numpy()[batch['real'].numpy()])
    return np.concatenate(result)


def assert_policy_identical(model, baseline):
    for name, value in baseline.state_dict().items():
        if not torch.equal(value.cpu(), model.state_dict()[name].cpu()):
            raise AssertionError(f'Probe changed policy/trunk buffer or weight: {name}')


def main():
    started = time.monotonic()
    config = yaml.safe_load(Path('configs/pokedecision-v1.yaml').read_text())
    torch.set_num_threads(min(4, config['train']['cpu_threads']))
    if not torch.cuda.is_available():
        raise RuntimeError('W4.4 requires CUDA')
    device = torch.device('cuda')
    root = Path(config['data_root'])
    if not (root / 'outcomes.json').exists():
        build_outcomes(root)
    train = load_train_tier('full', root)
    val = load_split('val', root)
    for split, arrays in [('train', train), ('val', val)]:
        arrays['outcomes'] = load_outcomes(root, split, arrays)
    stats = json.loads((root / 'feature_stats.json').read_text())
    encoder = json.loads((root / 'encoder.json').read_text())
    labels = np.repeat(val['outcomes'], np.diff(val['offsets']))
    train_rate = float(np.average(train['outcomes'], weights=np.diff(train['offsets'])))
    v1, _, _ = load_pokedecision('models/pokedecision-v1', device)
    loader = DataLoader(TrajectorySequences(val), batch_size=16,
                        collate_fn=pad_sequences)
    baseline = run_epoch(v1, loader, device, None, 1.0)
    references = {
        'constant_0.5': trajectory_metrics(np.full(len(labels), 0.5), labels,
                                            val['offsets']),
        'train_base_rate': trajectory_metrics(np.full(len(labels), train_rate),
                                              labels, val['offsets']),
        'train_step_base_rate': train_rate,
        'train_trajectory_base_rate': float(train['outcomes'].mean()),
        'v1_policy': baseline,
    }
    rating_p, rating_fit = rating_reference(
        train['ratings'], train['outcomes'], val['ratings']
    )
    references['rating_only'] = trajectory_metrics(
        np.repeat(rating_p, np.diff(val['offsets'])), labels, val['offsets']
    ) | {'fit': rating_fit}
    runs = []
    for mode, lr in [('probe', 0.001), ('joint', 0.0005)]:
        run_config = copy.deepcopy(config)
        destination = Path('checkpoints/w4-value') / mode
        run_config.update({
            'expected_parameters': 1300174, 'train_tier': 'full',
            'value_mode': mode, 'value_weight': 0.5,
            'init_checkpoint': 'models/pokedecision-v1/best.pt'
            if mode == 'probe' else None,
            'output_dir': str(destination),
        })
        run_config['model']['value_head'] = True
        run_config['train']['learning_rate'] = lr
        result = train_arch(run_config, 'gru', train, val, stats, encoder, destination)
        model, _, _ = load_pokedecision(destination, device)
        if mode == 'probe':
            assert_policy_identical(model, v1)
        value = trajectory_metrics(probabilities(model, val, device), labels,
                                   val['offsets'])
        policy = result['best_val']
        eligible = mode == 'probe' or policy['policy_loss'] <= baseline['loss'] + 0.01
        row = result | {
            'mode': mode, 'checkpoint_dir': str(destination), 'value': value,
            'eligible_for_config_D': eligible,
            'policy_loss_delta_v1': policy['policy_loss'] - baseline['loss'],
            'probe_policy_bit_identical': True if mode == 'probe' else None,
            'added_parameters': 385,
        }
        runs.append(row)
        (destination / 'metrics.json').write_text(json.dumps(row, indent=2) + '\n')
        del model
        torch.cuda.empty_cache()
    selected = min((r for r in runs if r['eligible_for_config_D']),
                   key=lambda r: r['value']['brier'])
    source = Path(selected['checkpoint_dir'])
    destination = Path('models/pokedecision-v1-value')
    destination.mkdir(parents=True, exist_ok=True)
    reason = ('Joint eligible only if W4 val policy loss <= v1 + 0.01; '
              'among eligible runs choose lowest W4 val Brier. '
              f"Selected {selected['mode']}.")
    for filename in ['best.pt', 'tokenizer.json', 'config.yaml', 'metrics.json']:
        shutil.copyfile(source / filename, destination / filename)
    metadata = yaml.safe_load((destination / 'config.yaml').read_text())
    metadata.update({'selected_from': str(source), 'selection_reason': reason,
                     'policy_version': 'pokedecision-v1-value'})
    (destination / 'config.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
    audit = {
        'encoder': 'Unmodified ObservationEncoder / DefaultObservationSpace-v1',
        'labels': 'WIN/LOSS in own POV replay filename only; outcomes never inputs',
        'same_battle_same_split': True,
        'test_splits_loaded': False,
        'terminal_state': 'w3.encode_trajectory drops final observation via [:-1]',
        'result_features': 'DefaultObservationSpace.state_to_obs never reads '
                           'battle_won or battle_lost; no end-of-battle token',
        'turn_bucket_semantics': 'Zero-based decision index, including forced switches; '
                                 'UniversalState does not retain simulator turn number',
        'last_state': 'Final observed pre-action state may be strategically decisive; '
                      'not terminal result. See final_step accuracy and Brier per run.',
        'turn0_leak_flags': {r['mode']: r['value']['first_step']['brier'] < 0.24
                             for r in runs},
        'turn0_material_threshold': 0.24,
        'turn0_investigation': 'Compare first_step metrics against rating-only baseline; '
                               'ratings are reference labels, not model features. '
                               'Initial species/HP/moves are legitimate public inputs.',
        'outcome_alignment': 'datasets/w4/outcomes.json includes ordered filename '
                             'POVs, NPZ alignment evidence and trajectory digests',
    }
    report = {
        'runs': runs, 'references': references, 'leakage_audit': audit,
        'selected_from': str(source), 'selection_reason': reason,
        'selection_policy_loss_limit': baseline['loss'] + 0.01,
        'value_used_for_action_selection': False,
        'paired_battle_impact': 'Deferred to W4.5; no win-rate improvement claim',
        'total_wall_time_s': time.monotonic() - started,
        'gpu': torch.cuda.get_device_name(),
    }
    path = Path('reports/w4_value_metrics.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'selected_from': str(source), 'wall_time_s': report['total_wall_time_s']}),
          flush=True)


if __name__ == '__main__':
    main()
