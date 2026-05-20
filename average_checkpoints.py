"""
Weight-space averaging of the last-N checkpoints (paper §2.6.2 "Model
Ensemble"):

    "We compute the average weights of the model of last several epochs to
     obtain the ensembled model."

Handles both:
  - Stage 1 contrastive checkpoints (key: 'model_state')
  - Stage 2 Siamese checkpoints     (keys: 'model_state' AND 'head_state')

By default it auto-detects every key whose value is a state_dict (dict of
tensors) and averages each one independently, so a single command processes
both the backbone and the Siamese head together.

Usage:
    # Stage 1 only
    python average_checkpoints.py \
        --ckpt_dir model/pet2022/ckpt_seresnet \
        --n 5 \
        --output model/pet2022/seresnet/pet2022_seresnet101_ibn_swa.pt

    # Stage 2 (averages BOTH model_state and head_state)
    python average_checkpoints.py \
        --ckpt_dir model/pet2022/ckpt_siamese \
        --n 5 \
        --output model/pet2022/seresnet/pet2022_seresnet101_ibn_stage2_swa.pt
"""

import argparse
import os
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, required=True,
                    help='directory containing ckpt_epoch_NNNN.pt files')
parser.add_argument('--n', type=int, default=5,
                    help='number of last checkpoints to average')
parser.add_argument('--output', type=str, required=True,
                    help='where to save the averaged checkpoint')
parser.add_argument('--keys', type=str, default='auto',
                    help='"auto" (default) averages every state-dict-shaped '
                         'key in the checkpoint. Or pass a comma-separated '
                         'list, e.g. "model_state,head_state".')
parser.add_argument('--include_int_buffers', action='store_true',
                    help='if set, also average integer buffers (BN running '
                         'counts etc). Off by default — those are kept as-is '
                         'from the last checkpoint.')
args = parser.parse_args()


def collect_checkpoints(ckpt_dir, n):
    files = []
    for fname in os.listdir(ckpt_dir):
        if fname.startswith('ckpt_epoch_') and fname.endswith('.pt'):
            try:
                ep = int(fname[len('ckpt_epoch_'):-len('.pt')])
                files.append((ep, os.path.join(ckpt_dir, fname)))
            except ValueError:
                continue
    if not files:
        raise FileNotFoundError(f'No ckpt_epoch_*.pt in {ckpt_dir}')
    files.sort()
    return files[-n:]


def is_state_dict(v):
    return (isinstance(v, dict)
            and len(v) > 0
            and all(isinstance(t, torch.Tensor) for t in v.values()))


def resolve_keys(first_state, requested):
    if requested == 'auto':
        keys = [k for k, v in first_state.items() if is_state_dict(v)]
        if not keys:
            raise KeyError(
                'No state-dict-shaped keys found in checkpoint; '
                f'available keys: {list(first_state.keys())}'
            )
        return keys
    keys = [k.strip() for k in requested.split(',') if k.strip()]
    missing = [k for k in keys if k not in first_state]
    if missing:
        raise KeyError(
            f'Requested key(s) {missing} not in checkpoint. '
            f'Available: {list(first_state.keys())}'
        )
    return keys


def average_one_key(selected, key, include_int_buffers):
    averaged = None
    last_sd = None
    for _, path in selected:
        state = torch.load(path, map_location='cpu', weights_only=False)
        sd = state[key]
        last_sd = sd
        if averaged is None:
            averaged = {k: v.detach().clone().float() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if not include_int_buffers and not v.is_floating_point():
                    continue
                averaged[k] += v.detach().float()

    n = len(selected)
    for k in list(averaged.keys()):
        if averaged[k].is_floating_point():
            averaged[k] = averaged[k] / n
            averaged[k] = averaged[k].to(last_sd[k].dtype)
        else:
            # int buffers we didn't sum — restore from last checkpoint
            averaged[k] = last_sd[k].clone()
    return averaged


def main():
    selected = collect_checkpoints(args.ckpt_dir, args.n)
    print(f'Averaging {len(selected)} checkpoints from {args.ckpt_dir}:')
    for ep, p in selected:
        print(f'  epoch {ep}: {p}')

    first = torch.load(selected[0][1], map_location='cpu', weights_only=False)
    keys = resolve_keys(first, args.keys)
    print(f'Averaging key(s): {keys}')

    out = {}
    for key in keys:
        print(f'  -> averaging "{key}"')
        out[key] = average_one_key(selected, key, args.include_int_buffers)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(out, args.output)
    print(f'Saved averaged checkpoint to: {args.output}')
    print(f'Top-level keys in output: {list(out.keys())}')


if __name__ == '__main__':
    main()
