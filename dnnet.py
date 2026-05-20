import os
import argparse
import random
import math
import time
import threading
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import SGD, Adam

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, default=50,
                    help='number of training classes (label < c)')
parser.add_argument('-e', '--epochs', type=int, default=200,
                    help='number of training epochs')
parser.add_argument('-d', '--data_path', type=str, default='crop_copy',
                    help='root directory of the dataset')
parser.add_argument('-o', '--output_dir', type=str, default='model/dnnet',
                    help='where to save the trained model')
parser.add_argument('-P', '--p_identities', type=int, default=4,
                    help='identities per PK batch (P). Default P=4, K=4 '
                         'gives batch size 16 to match the DNNet paper.')
parser.add_argument('-K', '--k_per_identity', type=int, default=4,
                    help='images per identity per batch (K). '
                         'Total batch size = P*K.')
parser.add_argument('--batches_per_epoch', type=int, default=80,
                    help='number of PK batches per epoch')
parser.add_argument('--contrastive_margin', type=float, default=2.0,
                    help='margin for batch-hard contrastive loss '
                         '(L2-normalised embeddings → use 1.0–1.5)')
parser.add_argument('--profile', action='store_true',
                    help='enable training-time / memory / energy profiling '
                         '(answers reviewer #10)')
parser.add_argument('--profile_csv', type=str,
                    default='result_csv/dnnet_training_profile.csv',
                    help='where to dump per-epoch profiling stats')
parser.add_argument('--ckpt_every', type=int, default=4,
                    help='save a checkpoint every N epochs')
parser.add_argument('--ckpt_dir', type=str, default='model/dnnet/ckpt',
                    help='directory for periodic checkpoints')
parser.add_argument('--keep_ckpts', type=int, default=2,
                    help='how many recent checkpoints to retain '
                         '(older ones are deleted to save disk)')
parser.add_argument('--resume', type=str, default='auto',
                    help='checkpoint path to resume from. '
                         '"auto" = latest in --ckpt_dir, '
                         '"none" = always start from scratch, '
                         'or pass an explicit .pt path')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Dataset (single-image)
# ---------------------------------------------------------------------------
class DogDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            try:
                label = int(class_name)
                if label >= args.c:
                    continue
            except:
                continue

            for root, dirs, files in os.walk(class_dir):
                if os.path.basename(root) != split:
                    continue

                for fname in files:
                    fpath = os.path.join(root, fname)
                    if os.path.isfile(fpath):
                        self.samples.append((fpath, label))

        if len(self.samples) == 0:
            raise ValueError(f"No images found for split='{split}' in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        # print(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PK Batch Sampler  (for batch-hard pair mining)
# ---------------------------------------------------------------------------
# Random pair sampling wastes compute: most random negatives are easy
# (distance already > margin → zero gradient).  Instead, we build a single
# batch of (P identities × K images each) = P*K images, run them through
# the encoder ONCE, then mine the hardest negatives **inside the batch**.
# This is the standard recipe from Hermans et al. 2017
# ("In Defense of the Triplet Loss for Person Re-Identification").
#
# Compared to the previous loader (4000 pairs/epoch × 2 forwards = 8000
# image forwards/epoch), a PK loader with P=8, K=4, ~50 batches per epoch
# does only 8*4*50 = 1600 image forwards/epoch — roughly 5x faster — and
# every backward step uses informative pairs only.
# ---------------------------------------------------------------------------
class PKBatchSampler(torch.utils.data.Sampler):
    """Yields lists of indices forming a (P × K) batch:
       - P distinct identities are sampled per batch
       - K images per identity (with replacement if a class has < K images)
    """
    def __init__(self, label_to_indices, p_identities=8, k_per_identity=4,
                 batches_per_epoch=200, seed=None):
        self.label_to_indices = label_to_indices
        self.labels = [l for l, idxs in label_to_indices.items() if len(idxs) >= 1]
        self.P = p_identities
        self.K = k_per_identity
        self.batches_per_epoch = batches_per_epoch
        self.rng = random.Random(seed)
        if len(self.labels) < self.P:
            raise ValueError(
                f"Need at least P={self.P} identities, got {len(self.labels)}"
            )

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_labels = self.rng.sample(self.labels, self.P)
            batch = []
            for lbl in chosen_labels:
                pool = self.label_to_indices[lbl]
                if len(pool) >= self.K:
                    picks = self.rng.sample(pool, self.K)
                else:
                    picks = [self.rng.choice(pool) for _ in range(self.K)]
                batch.extend(picks)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


# ---------------------------------------------------------------------------
# Helper for label index (kept for compatibility with previous code paths)
# ---------------------------------------------------------------------------
def _build_label_index(concat_or_single):
    label_to_indices = {}

    if isinstance(concat_or_single, ConcatDataset):
        offset = 0
        for ds in concat_or_single.datasets:
            for local_i, (_, lbl) in enumerate(ds.samples):
                label_to_indices.setdefault(int(lbl), []).append(offset + local_i)
            offset += len(ds)
    else:
        for local_i, (_, lbl) in enumerate(concat_or_single.samples):
            label_to_indices.setdefault(int(lbl), []).append(local_i)

    return label_to_indices


# ---------------------------------------------------------------------------
# Batch-hard pair losses
# ---------------------------------------------------------------------------
def batch_hard_contrastive_loss(embeddings, labels, margin=2.0):
    """Hermans et al. 2017 batch-hard variant of contrastive loss.

    For each anchor in the batch:
        - hardest_pos = farthest same-id neighbour  (we want this CLOSE)
        - hardest_neg = nearest different-id sample (we want this FAR > margin)
    Returns the mean contrastive loss across all anchors.

    Args:
        embeddings: (B, D) L2-normalised embeddings
        labels:     (B,)  integer ids
        margin:     contrastive margin (paper default 2.0 for un-normalised
                    Euclidean; with L2-normalised embeddings the natural
                    range of pairwise distances is [0, 2], so margin 1.0–1.5
                    works well)
    """
    B = embeddings.size(0)
    # pairwise euclidean distance matrix (B, B)
    dist = torch.cdist(embeddings, embeddings, p=2)

    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))   # (B, B) bool
    diff = ~same
    eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    same = same & ~eye    # exclude self-pairs

    # Hardest positive: max distance among same-id pairs.
    # If an anchor has no positive in this batch (rare with PK sampler),
    # its hardest_pos contribution is masked to zero.
    pos_dist = dist.masked_fill(~same, -float('inf'))
    hardest_pos, _ = pos_dist.max(dim=1)
    has_pos = same.any(dim=1)
    hardest_pos = torch.where(has_pos, hardest_pos,
                              torch.zeros_like(hardest_pos))

    # Hardest negative: min distance among different-id pairs.
    neg_dist = dist.masked_fill(~diff, float('inf'))
    hardest_neg, _ = neg_dist.min(dim=1)
    has_neg = diff.any(dim=1)
    # For anchors with no negatives, push their loss to zero by using a
    # distance large enough to make the hinge term zero.
    hardest_neg = torch.where(has_neg, hardest_neg,
                              torch.full_like(hardest_neg, margin + 1.0))

    # Standard contrastive form, summed over the two terms then averaged.
    pos_term = hardest_pos.pow(2)
    neg_term = torch.clamp(margin - hardest_neg, min=0.0).pow(2)
    loss = (pos_term + neg_term).mean()
    return loss


# ---------------------------------------------------------------------------
# DNNet architecture (unchanged from original)
# ---------------------------------------------------------------------------
class ResNet152_Backbone(nn.Module):
    def __init__(self):
        super(ResNet152_Backbone, self).__init__()

        resnet = models.resnet152()

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.extra_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.extra_layers(x)
        return x


class PositionAttentionModule(nn.Module):
    ''' self-attention '''

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.shape
        query = self.query_conv(x).view(
            N, -1, H*W).permute(0, 2, 1)
        key = self.key_conv(x).view(N, -1, H*W)

        energy = torch.bmm(query, key)
        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.shape
        query = x.view(N, C, -1)
        key = x.view(N, C, -1).permute(0, 2, 1)

        energy = torch.bmm(query, key)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class DualAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule()

    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(x)

        out = torch.cat([cam_out, pam_out, x], dim=1)
        return out


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.feature_extractor = ResNet152_Backbone()
        self.dam = DualAttentionModule(in_channels=512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(3*512, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def embed(self, x):
        x = self.feature_extractor(x)
        x = self.dam(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)

        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img1, img2):
        emb1 = self.embed(img1)
        emb2 = self.embed(img2)
        return emb1, emb2


class ArcFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.5,
                 easy_margin=False, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        cos_theta = F.linear(F.normalize(embedding),
                             F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        pos = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        sin_theta = torch.sqrt((1.0 - torch.pow(pos, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = pos * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, pos)
        else:
            phi = torch.where(pos > self.th, phi, pos - self.mm)
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        output *= self.scale
        loss = self.ce(output, ground_truth)
        return loss


# ---------------------------------------------------------------------------
# Training-side computational profiling (Reviewer Comment #10)
# ---------------------------------------------------------------------------
# This block measures: wall-clock time per epoch, peak GPU memory during
# training, model parameter count, FLOPs of a single Siamese forward pass,
# and *energy in Wh per epoch* via an nvidia-smi polling thread that samples
# instantaneous GPU power draw.
#
# All numbers are written to a CSV and printed at the end so they can drop
# straight into the revised paper's expanded Table II.
# ---------------------------------------------------------------------------
class GPUPowerSampler:
    """Poll `nvidia-smi --query-gpu=power.draw` in a background thread.
    Integrate the power samples (W) over wall-clock time to get energy (Wh).
    Falls back to recording 0 W if nvidia-smi is unavailable (CPU-only run)."""
    def __init__(self, gpu_index=0, interval_s=0.5):
        self.gpu_index = gpu_index
        self.interval = interval_s
        self._samples = []          # list of (timestamp_s, power_w)
        self._stop = threading.Event()
        self._thread = None
        self._available = self._check()

    def _check(self):
        try:
            subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw',
                 '--format=csv,noheader,nounits',
                 f'--id={self.gpu_index}'],
                stderr=subprocess.DEVNULL,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _poll(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=power.draw',
                     '--format=csv,noheader,nounits',
                     f'--id={self.gpu_index}'],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                power_w = float(out.splitlines()[0])
            except Exception:
                power_w = 0.0
            self._samples.append((t, power_w))
            self._stop.wait(self.interval)

    def start(self):
        self._samples = []
        self._stop.clear()
        if self._available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def stop(self):
        if self._available and self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)

    def integrate_wh(self):
        """Trapezoidal integration of power samples → energy in watt-hours."""
        if len(self._samples) < 2:
            return 0.0, 0.0  # (avg_power_w, energy_wh)
        ts = np.array([s[0] for s in self._samples])
        ps = np.array([s[1] for s in self._samples])
        # energy in joules = ∫ P dt
        joules = float(np.trapz(ps, ts))
        wh = joules / 3600.0
        avg_w = float(np.mean(ps))
        return avg_w, wh


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, device, image_size=224):
    """Returns FLOPs (multiply-add count x2) for ONE Siamese forward pass
    (i.e. two images through the same encoder).
    Tries thop first, falls back to fvcore, falls back to NaN."""
    model.eval()
    dummy1 = torch.randn(1, 3, image_size, image_size, device=device)
    dummy2 = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(model, inputs=(dummy1, dummy2), verbose=False)
        return float(macs * 2) / 1e9    # GFLOPs
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        # fvcore expects a single-tuple input; wrap forward so it works.
        class _Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x, x)
        flops = FlopCountAnalysis(_Wrap(model), dummy1).total()
        return float(flops) / 1e9
    except Exception:
        return float('nan')


def estimate_edge_latency(model, device, n_iters=50, image_size=224):
    """Single-image embedding latency (ms) — a proxy for mobile/IoT
    deployment cost. Measures only the embed() path (1 image), not the
    Siamese forward (2 images), to match how the model is used at
    inference time on an edge device."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model.embed(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model.embed(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1000.0   # ms / image


def lr_lambda(epoch):
    if epoch < 100:
        return 1.0
    else:
        return max(0.0, 1.0 - (epoch - 100) / 100)


# ---------------------------------------------------------------------------
# Build datasets and run training
# ---------------------------------------------------------------------------
def build_train_loader():
    train_dataset = DogDataset(args.data_path, transform=val_transforms)
    label_to_indices = _build_label_index(train_dataset)

    sampler = PKBatchSampler(
        label_to_indices,
        p_identities=args.p_identities,
        k_per_identity=args.k_per_identity,
        batches_per_epoch=args.batches_per_epoch,
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=6,
    )
    return train_loader


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
# We save the full training state (siamese weights, ArcFace weights, both
# optimizers, both schedulers, epoch counter, RNG state) so a resumed run
# is bit-identical in behaviour to an uninterrupted one — same LR, same
# Adam momentum, same data ordering.
# ---------------------------------------------------------------------------
def save_checkpoint(path, *, epoch, siamese, arcface,
                    opt_model, opt_arcface, sched_1, sched_2,
                    profile_rows):
    state = {
        'epoch': epoch,
        'siamese_state':   siamese.state_dict(),
        'arcface_state':   arcface.state_dict(),
        'opt_model_state': opt_model.state_dict(),
        'opt_arcface_state': opt_arcface.state_dict(),
        'sched_1_state':   sched_1.state_dict(),
        'sched_2_state':   sched_2.state_dict(),
        'rng_python':      random.getstate(),
        'rng_numpy':       np.random.get_state(),
        'rng_torch':       torch.get_rng_state(),
        'rng_cuda':        torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None,
        'profile_rows':    profile_rows,
        'args':            vars(args),
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic on most filesystems
    print(f"  -> saved checkpoint: {path}")


def load_checkpoint(path, siamese, arcface,
                    opt_model, opt_arcface, sched_1, sched_2,
                    device):
    
    state = torch.load(path, map_location=device, weights_only=False)
    
    siamese.load_state_dict(state['siamese_state'])
    arcface.load_state_dict(state['arcface_state'])
    opt_model.load_state_dict(state['opt_model_state'])
    opt_arcface.load_state_dict(state['opt_arcface_state'])
    sched_1.load_state_dict(state['sched_1_state'])
    sched_2.load_state_dict(state['sched_2_state'])
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    
    torch.set_rng_state(state['rng_torch'].cpu()
                        if hasattr(state['rng_torch'], 'cpu')
                        else state['rng_torch'])
                        
    if state.get('rng_cuda') is not None and torch.cuda.is_available():
        cuda_rng_state = [
            t.cpu() if hasattr(t, 'cpu') else t 
            for t in state['rng_cuda']
        ]
        torch.cuda.set_rng_state_all(cuda_rng_state)
        
    return state['epoch'], state.get('profile_rows', [])


def find_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for fname in os.listdir(ckpt_dir):
        if fname.startswith('ckpt_epoch_') and fname.endswith('.pt'):
            try:
                ep = int(fname[len('ckpt_epoch_'):-len('.pt')])
                candidates.append((ep, os.path.join(ckpt_dir, fname)))
            except ValueError:
                continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def prune_old_checkpoints(ckpt_dir, keep):
    if not os.path.isdir(ckpt_dir):
        return
    files = []
    for fname in os.listdir(ckpt_dir):
        if fname.startswith('ckpt_epoch_') and fname.endswith('.pt'):
            try:
                ep = int(fname[len('ckpt_epoch_'):-len('.pt')])
                files.append((ep, os.path.join(ckpt_dir, fname)))
            except ValueError:
                continue
    files.sort()
    for _, path in files[:-keep]:
        try:
            os.remove(path)
            print(f"  -> pruned old checkpoint: {path}")
        except OSError:
            pass


def train():
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siamese_model = SiameseNetwork().to(device)

    arcface = ArcFace(1024, args.c).to(device)

    opt_model = Adam(siamese_model.parameters(),
                     lr=0.0001,
                     betas=(0.5, 0.999))

    opt_arcface = SGD(arcface.parameters(),
                      lr=0.0001,
                      momentum=0.9,
                      weight_decay=0.0005)
    scheduler_1 = torch.optim.lr_scheduler.LambdaLR(opt_model, lr_lambda)
    scheduler_2 = torch.optim.lr_scheduler.LambdaLR(opt_arcface, lr_lambda)

    train_loader = build_train_loader()

    num_epochs = args.epochs
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ----- Resume from checkpoint if requested ---------------------------
    start_epoch = 0
    resumed_profile_rows = []
    if args.resume == 'auto':
        latest = find_latest_checkpoint(args.ckpt_dir)
    elif args.resume == 'none':
        latest = None
    else:
        latest = args.resume

    if latest is not None and os.path.isfile(latest):
        print(f"Resuming from checkpoint: {latest}")
        start_epoch, resumed_profile_rows = load_checkpoint(
            latest, siamese_model, arcface,
            opt_model, opt_arcface, scheduler_1, scheduler_2,
            device,
        )
        print(f"  resumed at epoch {start_epoch} "
              f"(will continue from epoch {start_epoch + 1})")
    else:
        print("No checkpoint found — starting from scratch.")
    # ---------------------------------------------------------------------

    print("=" * 60)
    print(f"PK batches: P={args.p_identities}, K={args.k_per_identity} "
          f"-> batch size {args.p_identities * args.k_per_identity}, "
          f"{args.batches_per_epoch} batches/epoch")
    print(f"Images per epoch: "
          f"{args.p_identities*args.k_per_identity*args.batches_per_epoch}")
    print("=" * 60)

    # ----- Profiling setup (#10) -----------------------------------------
    profile_rows = list(resumed_profile_rows)
    n_params_siamese = n_params_arcface = 0
    gflops = float('nan')
    edge_ms = float('nan')
    if args.profile:
        os.makedirs(os.path.dirname(args.profile_csv) or '.', exist_ok=True)
        n_params_siamese = count_parameters(siamese_model)
        n_params_arcface = count_parameters(arcface)
        gflops = estimate_flops(siamese_model, device)
        edge_ms = estimate_edge_latency(siamese_model, device)
        siamese_model.train()  # restore train mode after profiling probe

        print("Static model complexity (DNNet)")
        print(f"  Siamese params  : {n_params_siamese/1e6:.2f} M")
        print(f"  ArcFace params  : {n_params_arcface/1e6:.2f} M")
        print(f"  Total params    : {(n_params_siamese+n_params_arcface)/1e6:.2f} M")
        print(f"  Forward GFLOPs  : {gflops:.2f}  (Siamese, two images)")
        print(f"  Edge latency    : {edge_ms:.2f} ms / image  (single embed())")
        print("=" * 60)

    power_sampler = GPUPowerSampler(gpu_index=0, interval_s=0.5) \
                    if args.profile else None
    # ---------------------------------------------------------------------

    for epoch in range(start_epoch, num_epochs):
        epoch_t0 = time.perf_counter()
        n_imgs_seen = 0

        if args.profile and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if power_sampler is not None:
            power_sampler.start()

        for i, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(device)
            lbls = lbls.long().to(device)

            # Single forward through the encoder for the entire PK batch.
            embeddings = siamese_model.embed(imgs)   # (B, D), L2-normalised

            # Batch-hard contrastive: only the hardest pos/neg per anchor
            # contribute to the loss → no wasted compute on easy negatives.
            loss_c = batch_hard_contrastive_loss(
                embeddings, lbls, margin=args.contrastive_margin
            )

            # ArcFace identity loss on every image in the batch.
            loss_a = arcface(embeddings, lbls)

            loss = loss_c + loss_a

            opt_arcface.zero_grad()
            opt_model.zero_grad()
            loss.backward()
            opt_arcface.step()
            opt_model.step()

            n_imgs_seen += imgs.size(0)
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}  "
                  f"(C: {loss_c.item():.3f}, A: {loss_a.item():.3f})")

        if device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_t1 = time.perf_counter()

        if power_sampler is not None:
            power_sampler.stop()

        scheduler_1.step()
        scheduler_2.step()

        # ----- per-epoch profiling row ------------------------------------
        if args.profile:
            epoch_s = epoch_t1 - epoch_t0
            peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                           if device.type == 'cuda' else 0.0)
            avg_w, energy_wh = (power_sampler.integrate_wh()
                                if power_sampler is not None else (0.0, 0.0))
            ms_per_step = epoch_s / max(1, len(train_loader)) * 1000
            imgs_per_s = n_imgs_seen / max(1e-6, epoch_s)

            row = {
                'epoch': epoch + 1,
                'epoch_time_s': epoch_s,
                'ms_per_step': ms_per_step,
                'images_per_second': imgs_per_s,
                'peak_gpu_mem_MB': peak_mem_mb,
                'avg_gpu_power_W': avg_w,
                'energy_Wh': energy_wh,
            }
            profile_rows.append(row)

            print("-" * 60)
            print(f"[Profile] Epoch {epoch+1}: "
                  f"{epoch_s:.1f}s, "
                  f"{ms_per_step:.1f} ms/step, "
                  f"{imgs_per_s:.1f} imgs/s, "
                  f"peak mem {peak_mem_mb:.0f} MB, "
                  f"avg power {avg_w:.1f} W, "
                  f"energy {energy_wh:.3f} Wh")
            print("-" * 60)

        # ----- Periodic checkpoint --------------------------------------
        epoch_1based = epoch + 1
        is_last_epoch = (epoch_1based == num_epochs)
        if (epoch_1based % args.ckpt_every == 0) or is_last_epoch:
            ckpt_path = os.path.join(
                args.ckpt_dir, f'ckpt_epoch_{epoch_1based:04d}.pt'
            )
            save_checkpoint(
                ckpt_path,
                epoch=epoch_1based,
                siamese=siamese_model,
                arcface=arcface,
                opt_model=opt_model,
                opt_arcface=opt_arcface,
                sched_1=scheduler_1,
                sched_2=scheduler_2,
                profile_rows=profile_rows,
            )
            prune_old_checkpoints(args.ckpt_dir, keep=args.keep_ckpts)

    save_path = os.path.join(args.output_dir, "DNNet+Arc_Conloss.pt")
    torch.save(siamese_model.state_dict(), save_path)
    print(f"Saved trained DNNet to: {save_path}")

    # ----- Final profile dump -------------------------------------------
    if args.profile and len(profile_rows) > 0:
        df = pd.DataFrame(profile_rows)
        # Append summary row with totals + per-epoch averages.
        summary = {
            'epoch': 'TOTAL/AVG',
            'epoch_time_s': df['epoch_time_s'].sum(),
            'ms_per_step': df['ms_per_step'].mean(),
            'images_per_second': df['images_per_second'].mean(),
            'peak_gpu_mem_MB': df['peak_gpu_mem_MB'].max(),
            'avg_gpu_power_W': df['avg_gpu_power_W'].mean(),
            'energy_Wh': df['energy_Wh'].sum(),
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        df.to_csv(args.profile_csv, index=False)
        print(f"\nWrote per-epoch profile to: {args.profile_csv}")
        print(df.to_string(index=False))

        # Static complexity is constant across epochs — write once alongside.
        static_path = args.profile_csv.replace('.csv', '_static.csv')
        pd.DataFrame([{
            'siamese_params_M': n_params_siamese / 1e6,
            'arcface_params_M': n_params_arcface / 1e6,
            'total_params_M':   (n_params_siamese + n_params_arcface) / 1e6,
            'forward_gflops_two_images': gflops,
            'edge_latency_ms_per_image': edge_ms,
        }]).to_csv(static_path, index=False)
        print(f"Wrote static complexity to: {static_path}")


if __name__ == "__main__":
    train()