"""
Stage 2: Siamese Verification Branch (paper §2.5).

Loads the Stage 1 contrastive-trained model and adds a Siamese head:
    score = sigmoid( FC2( LeakyReLU( FC1( |v_i - v_j| ) ) ) )

Matches the released reference code structure
(pet-biometrics/fastreid/modeling/losses/oim.py :: OIMLoss_siameseoffline):
    fully_connect1 = Linear(embed_dim, hidden_dim)
    fully_connect2 = Linear(hidden_dim, 1)
    act            = LeakyReLU
The reference uses Linear(2048, 512); we keep the same 4x compression ratio
adapted to the Stage 1 output width (512 -> 128 by default). The paper's
Figure 1 schematic shows an extra BN, but the released training code does
not include one — we follow the code.

This file is configured to load the existing 512-d Stage 1 checkpoints
(``--embed_dim 512``). If you later retrain Stage 1 with a wider output,
pass ``--embed_dim`` to match.

Two modes (paper §2.5):
  - online:  random pos/neg pair from each batch (default)
  - offline: top-100 hard pairs from the instance-level memory

The instance-level memory from Stage 1 is reloaded so we can mine hard pairs
directly without needing to recompute embeddings from scratch.

Deviation from paper: the paper finetunes on training+validation in Stage 2.
We finetune on the training split only — our dataset is fully labeled with
no large unlabeled validation pool that would benefit from this step
(same reasoning as the Stage-1 pseudo-label deviation documented in
pet_biometric_2022.py).
"""

import os
import argparse
import random
import math
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Reuse all the components defined in pet_biometric_2022.py
from pet_biometric_2022 import (
    PetBiometricModel,
    DogDataset,
    PKBatchSampler,
    build_label_index,
    build_train_transform,
    build_val_transform,
    GPUPowerSampler,
    NORMALIZE,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, default=50)
parser.add_argument('-e', '--epochs', type=int, default=30,
                    help='Stage-2 epochs. Paper finetunes for far fewer '
                         'epochs than Stage 1 since features are already '
                         'discriminative.')
parser.add_argument('-d', '--data_path', type=str, default='crop_copy')
parser.add_argument('-o', '--output_dir', type=str, default='model/pet2022')
parser.add_argument('--backbone', type=str, default='seresnet101_ibn',
                    choices=['seresnet101_ibn', 'resnetXt101_ibn',
                             'resnetSt101'])
parser.add_argument('--stage1_ckpt', type=str, default=None,
                    help='path to a Stage-1 checkpoint or final .pt file '
                         '(required when running this file as main)')
parser.add_argument('--mode', type=str, default='online',
                    choices=['online', 'offline', 'combined'],
                    help='Paper §2.5 describes online OR offline. '
                         '"combined" is an extension that uses both.')
parser.add_argument('-P', '--p_identities', type=int, default=5,
                    help='identities per PK batch (P). Matches Stage 1: '
                         'P*K = 64 with K=4.')
parser.add_argument('-K', '--k_per_identity', type=int, default=4)
parser.add_argument('--batches_per_epoch', type=int, default=25)
parser.add_argument('--img_size', type=int, default=384,
                    help='Stage 2 trains at the higher resolution from '
                         'Stage 1 finetune')
parser.add_argument('--lr', type=float, default=3.5e-5,
                    help='Reference Stage2 BASE_LR=3.5e-5 (10x drop from '
                         'Stage 1 base).')
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--embed_dim', type=int, default=2048,
                    help='Embedding dim. Default matches the 2048-d '
                         'Stage 1 produced by --finetune_from in '
                         'pet_biometric_2022.py (EMBEDDING_DIM=0 layout). '
                         'Use --embed_dim 512 if you really want to load '
                         'an old un-finetuned 512-d checkpoint.')
parser.add_argument('--aug', type=str, default='autoaug_phase2',
                    choices=['none', 'autoaug', 'autoaug_phase2'])
parser.add_argument('--freeze_backbone', action='store_true',
                    help='if set, only the Siamese head is trained')
parser.add_argument('--mem_momentum', type=float, default=0.5,
                    help='momentum for refreshing the ICL memory bank during '
                         'Stage 2 fine-tuning (matches Stage 1 ICL default).')
parser.add_argument('--offline_top_k', type=int, default=100,
                    help='Paper §2.5: top-K hard pool size for offline mining.')
parser.add_argument('--ckpt_every', type=int, default=4)
parser.add_argument('--run', type=int, default=1,
                    help='Run number, used as the trailing suffix when '
                         '--ckpt_dir is left at its default.')
parser.add_argument('--ckpt_dir', type=str, default=None,
                    help='If unset, defaults to '
                         'model/pet2022/ckpt_siamese_{backbone_short}_{run}, '
                         'e.g. model/pet2022/ckpt_siamese_seresnet_1.')
parser.add_argument('--keep_ckpts', type=int, default=2)
parser.add_argument('--resume', type=str, default='auto')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--profile_csv', type=str,
                    default='result_csv/pet2022_siamese_profile.csv')
args, _ = parser.parse_known_args()

_BACKBONE_SHORT = {
    'seresnet101_ibn': 'seresnet',
    'resnext101_ibn':  'resnext',
    'resnest101':      'resnest',
}
if args.ckpt_dir is None:
    short = _BACKBONE_SHORT.get(args.backbone, args.backbone)
    args.ckpt_dir = f'model/pet2022/ckpt_siamese_{short}_{args.run}'


# ---------------------------------------------------------------------------
# Siamese head (paper §2.5)
# ---------------------------------------------------------------------------
class SiameseHead(nn.Module):
    """Siamese verification head, matching the reference implementation
    in ``OIMLoss_siameseoffline`` (pet-biometrics repo):

        h     = LeakyReLU( Linear(embed_dim, hidden_dim)(|v_i - v_j|) )
        logit = Linear(hidden_dim, 1)(h)

    The reference is exactly ``Linear(2048, 512)`` followed by
    ``Linear(512, 1)``. With ``hidden_dim=None`` we auto-pick
    ``hidden_dim = embed_dim // 4`` so embed_dim=2048 -> hidden=512
    reproduces the reference numerically.

    Returns a raw logit; pair with BCEWithLogitsLoss.
    """
    def __init__(self, embed_dim=2048, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(embed_dim // 4, 64)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, vi, vj):
        diff = (vi - vj).abs()
        h = self.act(self.fc1(diff))
        return self.fc2(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Pair mining
# ---------------------------------------------------------------------------
def mine_pairs_online(embeddings, labels):
    """For each anchor in the batch, pick one positive (random same-id) and
    one negative (random diff-id). Returns three (B,) tensors:
        (anchor_idx, partner_idx, pair_label_in_{0,1})
    """
    B = embeddings.size(0)
    pairs = []  # list of (i, j, y)
    for i in range(B):
        same = (labels == labels[i]).clone()
        same[i] = False
        diff = (labels != labels[i])
        if same.sum() > 0:
            j = same.nonzero(as_tuple=False).view(-1)
            pairs.append((i, int(j[torch.randint(len(j), (1,))]), 1))
        if diff.sum() > 0:
            j = diff.nonzero(as_tuple=False).view(-1)
            pairs.append((i, int(j[torch.randint(len(j), (1,))]), 0))
    if not pairs:
        return None, None, None
    ai = torch.tensor([p[0] for p in pairs], device=embeddings.device)
    bj = torch.tensor([p[1] for p in pairs], device=embeddings.device)
    yy = torch.tensor([p[2] for p in pairs], device=embeddings.device,
                      dtype=torch.float32)
    return ai, bj, yy


def mine_pairs_offline(embeddings, labels, memory, memory_labels, top_k=100):
    """Paper §2.5 offline mining (strict).

    For each anchor in the batch:
      1. Find top-K most similar memory entries (the hard pool).
      2. RANDOMLY select one positive (same id) and one negative (diff id)
         from that pool.
      3. Partner feature vj/vk is taken DIRECTLY from the memory bank
         (paper writes 'vj ∈ R^(1×d)' as a selected memory entry, not as
         a re-extracted feature).

    Returns (vi, vj, y) tensors ready to feed the Siamese head.
    """
    sim = embeddings @ memory.t()                         # (B, N)
    vi_list, vj_list, y_list = [], [], []

    for i in range(embeddings.size(0)):
        scores = sim[i]
        _, top_idx = torch.topk(scores, min(top_k, scores.numel()))

        same = (memory_labels[top_idx] == labels[i])
        diff = ~same & (memory_labels[top_idx] >= 0)

        if same.any():
            pos_pool = top_idx[same]
            j_pos = pos_pool[torch.randint(len(pos_pool), (1,)).item()]
            vi_list.append(embeddings[i])
            vj_list.append(memory[j_pos])
            y_list.append(1.0)

        if diff.any():
            neg_pool = top_idx[diff]
            j_neg = neg_pool[torch.randint(len(neg_pool), (1,)).item()]
            vi_list.append(embeddings[i])
            vj_list.append(memory[j_neg])
            y_list.append(0.0)

    if not vi_list:
        return None, None, None
    vi = torch.stack(vi_list)
    vj = torch.stack(vj_list)
    y = torch.tensor(y_list, device=embeddings.device, dtype=torch.float32)
    return vi, vj, y


# ---------------------------------------------------------------------------
# Checkpointing (Siamese stage)
# ---------------------------------------------------------------------------
def save_ckpt(path, *, epoch, model, head, optimizer, profile_rows):
    state = {
        'epoch': epoch,
        'model_state':  model.state_dict(),
        'head_state':   head.state_dict(),
        'optim_state':  optimizer.state_dict(),
        'rng_python':   random.getstate(),
        'rng_numpy':    np.random.get_state(),
        'rng_torch':    torch.get_rng_state(),
        'rng_cuda':     torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None,
        'profile_rows': profile_rows,
        'args':         vars(args),
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)
    print(f'  -> saved Siamese checkpoint: {path}')


def load_ckpt(path, model, head, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    head.load_state_dict(state['head_state'])
    optimizer.load_state_dict(state['optim_state'])
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'].cpu()
                        if hasattr(state['rng_torch'], 'cpu')
                        else state['rng_torch'])
    if state.get('rng_cuda') is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        except TypeError:
            # Stale checkpoint stored rng_cuda in a format the current
            # torch can't ingest directly; non-critical, skip.
            pass
    return state['epoch'], state.get('profile_rows', [])


def find_latest(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    cs = []
    for f in os.listdir(ckpt_dir):
        if f.startswith('ckpt_epoch_') and f.endswith('.pt'):
            try:
                ep = int(f[len('ckpt_epoch_'):-len('.pt')])
                cs.append((ep, os.path.join(ckpt_dir, f)))
            except ValueError:
                continue
    if not cs:
        return None
    cs.sort()
    return cs[-1][1]


def prune(ckpt_dir, keep):
    if not os.path.isdir(ckpt_dir):
        return
    fs = []
    for f in os.listdir(ckpt_dir):
        if f.startswith('ckpt_epoch_') and f.endswith('.pt'):
            try:
                ep = int(f[len('ckpt_epoch_'):-len('.pt')])
                fs.append((ep, os.path.join(ckpt_dir, f)))
            except ValueError:
                continue
    fs.sort()
    for _, p in fs[:-keep]:
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def build_loader(img_size):
    transform = build_train_transform(img_size, args.aug)
    dataset = DogDataset(args.data_path, transform=transform)
    label_to_indices = build_label_index(dataset)
    sampler = PKBatchSampler(label_to_indices,
                             p_identities=args.p_identities,
                             k_per_identity=args.k_per_identity,
                             batches_per_epoch=args.batches_per_epoch,
                             seed=42)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=4, pin_memory=True)
    return loader, dataset


def train():
    if args.stage1_ckpt is None:
        raise SystemExit('error: --stage1_ckpt is required to run Stage 2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Build model and load Stage 1 weights ---
    model = PetBiometricModel(backbone_name=args.backbone,
                              embed_dim=args.embed_dim,
                              pretrained=False).to(device)

    def strip_profiling(sd):
        return {k: v for k, v in sd.items()
                if 'total_ops' not in k and 'total_params' not in k}

    print(f'Loading Stage-1 weights from: {args.stage1_ckpt}')
    state = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
    if 'model_state' in state:
        model.load_state_dict(strip_profiling(state['model_state']))
        # Try to recover the instance memory for offline pair mining.
        memory = None
        memory_labels = None
        if args.mode in ('offline', 'combined') and 'icl_state' in state:
            memory = state['icl_state']['memory'].to(device)
            memory_labels = state['icl_state']['memory_labels'].to(device)
            print(f'  recovered instance memory: '
                  f'{memory.shape[0]} entries x {memory.shape[1]}-d')
        elif args.mode in ('offline', 'combined'):
            print('WARNING: --mode offline/combined requested but Stage-1 '
                  'checkpoint has no ICL memory. Falling back to online.')
            args.mode = 'online'
    else:
        # final flat state_dict
        model.load_state_dict(strip_profiling(state))
        memory, memory_labels = None, None
        if args.mode in ('offline', 'combined'):
            print('WARNING: --mode offline/combined requested but Stage-1 file '
                  'is a flat state_dict without ICL memory. Falling back to '
                  'online mining.')
            args.mode = 'online'

    head = SiameseHead(embed_dim=args.embed_dim).to(device)

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        params = list(head.parameters())
        print('Backbone frozen — training Siamese head only.')
    else:
        params = list(model.parameters()) + list(head.parameters())

    optimizer = Adam(params, lr=args.lr, weight_decay=args.wd)
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # --- Resume Stage-2 if a checkpoint exists ---
    start_epoch = 0
    profile_rows = []
    if args.resume == 'auto':
        latest = find_latest(args.ckpt_dir)
    elif args.resume == 'none':
        latest = None
    else:
        latest = args.resume
    if latest is not None and os.path.isfile(latest):
        print(f'Resuming Stage-2 from: {latest}')
        start_epoch, profile_rows = load_ckpt(latest, model, head,
                                              optimizer, device)

    print('=' * 60)
    # print(args.mode)
    mode_desc = {'online': 'online only', 'offline': 'offline only',
                 'combined': 'online + offline'}[args.mode]
    print(f'Stage 2 (Siamese): backbone={args.backbone}, '
          f'mode={args.mode} ({mode_desc}), '
          f'img_size={args.img_size}, freeze_backbone={args.freeze_backbone}')
    print('=' * 60)

    train_loader, _ = build_loader(args.img_size)
    power_sampler = GPUPowerSampler() if args.profile else None

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.perf_counter()
        n_imgs = 0
        if args.profile and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if power_sampler is not None:
            power_sampler.start()

        for i, (imgs, lbls, idxs) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.long().to(device)

            if args.freeze_backbone:
                with torch.no_grad():
                    embeddings = model(imgs)
            else:
                embeddings = model(imgs)

            # Pair selection (paper §2.5).
            if args.mode == 'online':
                ai, bj, yy = mine_pairs_online(embeddings, lbls)
                if ai is None:
                    continue
                vi = embeddings[ai]
                vj = embeddings[bj]
            elif args.mode == 'offline':
                vi, vj, yy = mine_pairs_offline(
                    embeddings, lbls, memory, memory_labels,
                    top_k=args.offline_top_k,
                )
                if vi is None:
                    continue
            else:  # combined
                ai, bj, yy_on = mine_pairs_online(embeddings, lbls)
                vi_off, vj_off, yy_off = mine_pairs_offline(
                    embeddings, lbls, memory, memory_labels,
                    top_k=args.offline_top_k,
                )
                has_online = ai is not None
                has_offline = vi_off is not None
                if not has_online and not has_offline:
                    continue
                parts_vi, parts_vj, parts_y = [], [], []
                if has_online:
                    parts_vi.append(embeddings[ai])
                    parts_vj.append(embeddings[bj])
                    parts_y.append(yy_on)
                if has_offline:
                    parts_vi.append(vi_off)
                    parts_vj.append(vj_off)
                    parts_y.append(yy_off)
                vi = torch.cat(parts_vi)
                vj = torch.cat(parts_vj)
                yy = torch.cat(parts_y)

            # Refresh memory for the current batch's anchor indices too,
            # so the bank tracks the fine-tuned backbone.
            if memory is not None:
                with torch.no_grad():
                    idxs_dev = idxs.to(device)
                    memory[idxs_dev] = F.normalize(
                        args.mem_momentum * memory[idxs_dev]
                        + (1.0 - args.mem_momentum) * embeddings.detach(),
                        dim=1,
                    )
                    memory_labels[idxs_dev] = lbls

            logits = head(vi, vj)
            loss = bce(logits, yy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_imgs += imgs.size(0)
            print(f'Stage2 Epoch [{epoch+1}/{args.epochs}], '
                  f'Step [{i+1}/{len(train_loader)}], '
                  f'BCE: {loss.item():.4f}, '
                  f'pairs: {yy.numel()}')

        if device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_t1 = time.perf_counter()
        if power_sampler is not None:
            power_sampler.stop()

        if args.profile:
            epoch_s = epoch_t1 - epoch_t0
            peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                       if device.type == 'cuda' else 0.0)
            avg_w, energy_wh = (power_sampler.integrate_wh()
                                if power_sampler is not None else (0.0, 0.0))
            profile_rows.append({
                'stage': 'siamese',
                'epoch': epoch + 1,
                'epoch_time_s': epoch_s,
                'images_per_second': n_imgs / max(1e-6, epoch_s),
                'peak_gpu_mem_MB': peak_mb,
                'avg_gpu_power_W': avg_w,
                'energy_Wh': energy_wh,
            })
            print(f'[Profile] {epoch_s:.1f}s, peak {peak_mb:.0f} MB, '
                  f'energy {energy_wh:.3f} Wh')

        # Periodic checkpoint
        ep1 = epoch + 1
        if ep1 % args.ckpt_every == 0 or ep1 == args.epochs:
            ckpt_path = os.path.join(args.ckpt_dir,
                                     f'ckpt_epoch_{ep1:04d}.pt')
            save_ckpt(ckpt_path, epoch=ep1, model=model, head=head,
                      optimizer=optimizer, profile_rows=profile_rows)
            prune(args.ckpt_dir, keep=args.keep_ckpts)

    # Save final
    final_path = os.path.join(args.output_dir,
                              f'pet2022_{args.backbone}_stage2.pt')
    torch.save({'model_state': model.state_dict(),
                'head_state': head.state_dict()}, final_path)
    print(f'Saved final Stage-2 model + Siamese head to: {final_path}')

    if args.profile and profile_rows:
        os.makedirs(os.path.dirname(args.profile_csv) or '.', exist_ok=True)
        pd.DataFrame(profile_rows).to_csv(args.profile_csv, index=False)


if __name__ == '__main__':
    train()
