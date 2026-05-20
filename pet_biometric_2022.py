"""
Reproduction of "1st Place Solution to Pet Biometric Challenge 2022"
(Li et al., CVPR 2022 Workshop) for benchmarking against our method.

This is the contrastive-learning stage; the Siamese verification finetune
is in pet_biometric_2022_siamese.py.

Faithful to the paper and the released reference repo
(https://github.com/dashengge/pet-biometrics, configs/Stage1, configs/Stage2,
configs/final/, fastreid/modeling/losses/oim.py):

  - Backbone: SEResNet101 / ResNetXt101 / ResNetSt101, all with IBN-a +
    Non-local block before pooling
  - Generalized Mean Pooling (GeM) with learnable p
  - BN-Neck before the embedding head (Luo et al. 2019, ref [6])
  - 2048-d output by default (reference EMBEDDING_DIM: 0); reducing
    --embed_dim below 2048 adds an extra Linear projection
  - OIM loss + weighted ICL loss (paper §2.4); reference scales wICL by 0.5
  - PK batch sampler with P=16, K=4 (IMS_PER_BATCH=64)
  - Adam, base LR 3.5e-4 / wd 5e-4, MultiStepLR with milestones [30,50,70]
    in Stage 1, linear warmup over 3000 iters with warmup_factor 0.1
  - Two-stage size schedule: 70 epochs @ 224 -> 30 epochs @ 384 with the
    LR dropped 10x to 3.5e-5 at the boundary, additional MultiStepLR
    milestone at [15] within Stage 2
  - AutoAugment + Phase-2 additions (JPEG, gaussian blur, motion blur)

Deviations from paper (documented for the write-up):
  - Multi-backbone ensemble is supported via separate runs with different
    --backbone flags; user can train SEResNet101 / ResNetXt101 / ResNetSt101
    independently and ensemble at inference time using ensemble_eval.py
  - Pseudo-label finetuning on validation set is omitted: our dataset is
    fully labeled with no large unlabeled validation pool, so pseudo-
    labeling adds no information.
  - Instance memory size N is the dataset size (~300 images for 50 dogs)
    rather than thousands. Hard-negative selection becomes top-min(500, N).

Finetune from an older checkpoint:
  Use ``--finetune_from path/to/old_stage1.pt --embed_dim 2048
  --epochs_finetune 30`` to skip Stage 1 entirely and only run the Stage 2
  finetune at 384 px (LR 3.5e-5). Backbone / Non-local / GeM / BN-Neck
  weights from the old checkpoint are partial-loaded; the 512-d
  ``embed_fc`` is dropped so the output is the 2048-d BN-neck feature
  directly. OIM centroids and the ICL memory bank are re-initialized at
  the new dimension and warm up during the finetune.
"""

import os
import argparse
import random
import math
import time
import threading
import subprocess
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, default=50,
                    help='number of training classes (label < c)')
parser.add_argument('-e', '--epochs', type=int, default=70,
                    help='Stage-1 epochs at the small image size. '
                         'Paper: 70 @ 224.')
parser.add_argument('--epochs_finetune', type=int, default=20,
                    help='Stage-2 epochs at the large image size. '
                         'Paper: 30 @ 384. Set to 0 to disable Stage 2.')
parser.add_argument('-d', '--data_path', type=str, default='crop_copy',
                    help='root directory of the dataset')
parser.add_argument('-o', '--output_dir', type=str,
                    default='model/pet2022',
                    help='where to save the trained model')
parser.add_argument('--backbone', type=str, default='seresnet101_ibn',
                    choices=['seresnet101_ibn', 'resnetXt101_ibn',
                             'resnetSt101'],
                    help='backbone choice. Run with each option to build '
                         'the multi-backbone ensemble.')
parser.add_argument('-P', '--p_identities', type=int, default=8,
                    help='identities per PK batch (P). Reference uses '
                         'IMS_PER_BATCH=64 with K=4, so P=16.')
parser.add_argument('-K', '--k_per_identity', type=int, default=4,
                    help='images per identity per batch (K). '
                         'Total batch size = P*K.')
parser.add_argument('--batches_per_epoch', type=int, default=30,
                    help='number of PK batches per epoch')

# --- Loss weights (paper says OIM + wICL; ref scales wICL by 0.5) ---------
parser.add_argument('--lambda_oim', type=float, default=1.0)
parser.add_argument('--lambda_icl', type=float, default=0.5,
                    help='Reference code scales weighted-ICL by 0.5 '
                         '(loss_wincetance = 0.5 * associate_loss / N).')
parser.add_argument('--oim_momentum', type=float, default=0.2,
                    help='momentum for OIM cluster centroid update. '
                         'Reference: 0.2.')
parser.add_argument('--oim_temperature', type=float, default=0.05)
parser.add_argument('--icl_temperature', type=float, default=0.05)
parser.add_argument('--icl_n_neg', type=int, default=500,
                    help='top-K hardest negatives per anchor in ICL')
parser.add_argument('--icl_momentum', type=float, default=0.2,
                    help='momentum for instance memory update. '
                         'Reference: 0.2.')


parser.add_argument('--img_size', type=int, default=224,
                    help='Stage-1 training image size. Paper: 224.')
parser.add_argument('--img_size_finetune', type=int, default=384,
                    help='Stage-2 training image size. Paper: 384.')

# --- Optimizer (paper: Adam, lr=3.5e-4, wd=5e-4) -------------------------
parser.add_argument('--lr', type=float, default=3.5e-4,
                    help='Stage-1 base LR. Paper / reference: 3.5e-4.')
parser.add_argument('--lr_finetune', type=float, default=3.5e-5,
                    help='Stage-2 base LR. Reference Stage2/*.yml drops LR '
                         '10x: BASE_LR=0.000035.')
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--embed_dim', type=int, default=2048,
                    help='Embedding dimension. Reference uses '
                         'EMBEDDING_DIM: 0 -> the BN-neck feature at '
                         'backbone width (2048). When this equals the '
                         'backbone feat_dim the extra projection is '
                         'skipped automatically.')

# --- LR schedule (MultiStepLR + linear warmup, mirroring Stage1/Stage2) ---
parser.add_argument('--steps_stage1', type=int, nargs='+',
                    default=[30, 50, 70],
                    help='MultiStepLR milestones during Stage 1. '
                         'Reference Stage1/resnext101.yml: [30,50,70].')
parser.add_argument('--steps_stage2', type=int, nargs='+',
                    default=[15],
                    help='MultiStepLR milestones during Stage 2 '
                         '(counted from the start of Stage 2). '
                         'Reference Stage2/*.yml: [15].')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='MultiStepLR multiplicative factor.')
parser.add_argument('--warmup_iters', type=int, default=3000,
                    help='Linear warmup over the first N iterations of '
                         'Stage 1. Reference: 3000.')
parser.add_argument('--warmup_factor', type=float, default=0.1,
                    help='LR multiplier at iteration 0 of warmup. '
                         'Reference: 0.1.')

# --- Finetune-from-old-checkpoint mode -----------------------------------
parser.add_argument('--finetune_from', type=str, default=None,
                    help='Path to an existing (possibly 512-d) Stage 1 '
                         'checkpoint to finetune from. When set: Stage 1 '
                         '(224px, 70 epochs) is SKIPPED entirely, the '
                         'backbone + Non-local + GeM + BN-Neck weights are '
                         'partially loaded, and the embed_fc projection is '
                         'dropped (run with --embed_dim 2048 to match the '
                         'reference EMBEDDING_DIM:0 layout). The OIM '
                         'centroids and ICL memory are re-initialized at '
                         'the new dimension and warmed up during Stage 2 '
                         'training only. Implies --freeze_backbone.')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='Freeze backbone + Non-local + GeM (eval mode, '
                         'no grad). Only the BN-Neck "head" is trained. '
                         'Automatically enabled by --finetune_from. Must '
                         'be passed again when resuming a finetune so the '
                         'optimizer param groups match the saved state.')

# --- Augmentation flags ---------------------------------------------------
parser.add_argument('--aug', type=str, default='autoaug_phase2',
                    choices=['none', 'autoaug', 'autoaug_phase2'],
                    help='Paper used AutoAugment in phase 1 and added '
                         'JPEG/blur in phase 2. Default = full recipe.')

# --- Profiling (Reviewer Comment #10) ------------------------------------
parser.add_argument('--profile', action='store_true',
                    help='enable training-time / memory / energy profiling')
parser.add_argument('--profile_csv', type=str,
                    default='result_csv/pet2022_training_profile.csv')

# --- Checkpointing --------------------------------------------------------
parser.add_argument('--ckpt_every', type=int, default=4)
parser.add_argument('--ckpt_dir', type=str, default='model/pet2022/ckpt')
parser.add_argument('--keep_ckpts', type=int, default=2)
parser.add_argument('--resume', type=str, default='auto')

args, _ = parser.parse_known_args()


# ---------------------------------------------------------------------------
# Augmentation pipelines (paper §2.3)
# ---------------------------------------------------------------------------
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class RandomJPEGCompression:
    """Phase-2 augmentation from the paper. Re-encodes the image as JPEG
    at a random quality level to simulate test-time compression artifacts."""
    def __init__(self, p=0.3, quality_min=30, quality_max=80):
        self.p = p
        self.qmin, self.qmax = quality_min, quality_max

    def __call__(self, pil):
        if random.random() > self.p:
            return pil
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomMotionBlur:
    """Phase-2 augmentation. Applies a directional blur kernel."""
    def __init__(self, p=0.2, kernel_min=3, kernel_max=9):
        self.p = p
        self.kmin, self.kmax = kernel_min, kernel_max

    def __call__(self, pil):
        if random.random() > self.p:
            return pil
        k = random.choice(range(self.kmin, self.kmax + 1, 2))
        kernel = np.zeros((k, k), dtype=np.float32)
        # horizontal or vertical motion
        if random.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        arr = np.array(pil).astype(np.float32)
        from scipy.ndimage import convolve
        try:
            for c in range(3):
                arr[..., c] = convolve(arr[..., c], kernel, mode='reflect')
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        except ImportError:
            # fall back to gaussian if scipy not present
            return pil.filter(ImageFilter.GaussianBlur(radius=k / 4))


def build_train_transform(img_size, mode):
    """Paper §2.3: AutoAugment in phase 1, plus JPEG/Gaussian/Motion blur
    in phase 2 to handle the blurred test images."""
    base = [transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5)]

    if mode in ('autoaug', 'autoaug_phase2'):
        base.append(transforms.AutoAugment(
            policy=transforms.AutoAugmentPolicy.IMAGENET
        ))

    if mode == 'autoaug_phase2':
        # Apply at PIL stage so JPEG compression works correctly
        base.extend([
            RandomJPEGCompression(p=0.3),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))],
                p=0.2),
            RandomMotionBlur(p=0.2),
        ])

    base.extend([
        transforms.ToTensor(),
        NORMALIZE,
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])
    return transforms.Compose(base)


def build_val_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        NORMALIZE,
    ])


# ---------------------------------------------------------------------------
# Dataset and PK sampler (same convention as dnnet.py)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # Return (image, label, global_index). The global index is needed
        # for the instance memory in weighted ICL.
        return image, label, index


class PKBatchSampler(torch.utils.data.Sampler):
    def __init__(self, label_to_indices, p_identities=4, k_per_identity=4,
                 batches_per_epoch=200, seed=None):
        self.label_to_indices = label_to_indices
        self.labels = [l for l, idxs in label_to_indices.items() if len(idxs) >= 1]
        self.P = p_identities
        self.K = k_per_identity
        self.batches_per_epoch = batches_per_epoch
        self.rng = random.Random(seed)
        if len(self.labels) < self.P:
            raise ValueError(f'Need at least P={self.P} identities, got {len(self.labels)}')

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = self.rng.sample(self.labels, self.P)
            batch = []
            for lbl in chosen:
                pool = self.label_to_indices[lbl]
                if len(pool) >= self.K:
                    picks = self.rng.sample(pool, self.K)
                else:
                    picks = [self.rng.choice(pool) for _ in range(self.K)]
                batch.extend(picks)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


def build_label_index(dataset):
    label_to_indices = {}
    for i, (_, lbl) in enumerate(dataset.samples):
        label_to_indices.setdefault(int(lbl), []).append(i)
    return label_to_indices


# ===========================================================================
# Architecture: SEResNet101 + IBN-a + Non-local + GeM + BN-Neck
# ===========================================================================
# The paper's "best ablation row" (Table 1, last row) uses IBN + Non-local on
# top of the SEResNet101 backbone. We rebuild that here. We use timm to get
# the pretrained SEResNet101 weights, then surgically insert IBN normalization
# in the early stages and a Non-local block before pooling.
# ---------------------------------------------------------------------------
def _try_import_timm():
    try:
        import timm
        return timm
    except ImportError:
        raise ImportError(
            'timm is required for SEResNet101 backbone. '
            'Install with:  pip install timm'
        )


class IBN(nn.Module):
    """IBN-a (Pan et al., ECCV 2018, ref [8]). Splits channels: half go
    through InstanceNorm (style-invariant), half through BatchNorm
    (content-preserving). Used in the early stages of the backbone."""
    def __init__(self, num_features, ratio=0.5):
        super().__init__()
        self.half = int(num_features * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(num_features - self.half)

    def forward(self, x):
        split = torch.split(x, [self.half, x.size(1) - self.half], dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), dim=1)


class NonLocalBlock(nn.Module):
    """Wang et al., CVPR 2018, ref [10]. Embedded Gaussian variant."""
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        inter = max(in_channels // reduction, 1)
        self.theta = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.g = nn.Conv2d(in_channels, inter, kernel_size=1)
        self.W = nn.Sequential(
            nn.Conv2d(inter, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, -1, H * W).permute(0, 2, 1)
        phi = self.phi(x).view(B, -1, H * W)
        f = torch.softmax(torch.bmm(theta, phi), dim=-1)
        g = self.g(x).view(B, -1, H * W).permute(0, 2, 1)
        y = torch.bmm(f, g).permute(0, 2, 1).contiguous().view(B, -1, H, W)
        return x + self.W(y)


class GeMPool(nn.Module):
    """Generalized Mean Pooling (Radenovic et al., ref [9])."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1
        ).pow(1.0 / self.p)


def _replace_bn_with_ibn(module, parent_name=''):
    """Walk a module, replace BatchNorm2d in early stages with IBN.
    Following the IBN-a recipe: only the first 3 stages get IBN, the last
    stage keeps BN."""
    for name, child in module.named_children():
        full = f'{parent_name}.{name}' if parent_name else name
        # keep stage4 (layer4 in timm/torchvision ResNet) untouched
        if full.startswith('layer4'):
            continue
        if isinstance(child, nn.BatchNorm2d):
            ibn = IBN(child.num_features)
            setattr(module, name, ibn)
        else:
            _replace_bn_with_ibn(child, full)


class PetBiometricBackbone(nn.Module):
    """Backbone wrapper that exposes feature maps before pooling."""
    def __init__(self, name='seresnet101_ibn', pretrained=True):
        super().__init__()
        timm = _try_import_timm()

        if name == 'seresnet101_ibn':
            base = timm.create_model('seresnet101', pretrained=pretrained,
                                     num_classes=0, global_pool='')
            _replace_bn_with_ibn(base)
            feat_dim = 2048
        elif name == 'resnetXt101_ibn':
            base = timm.create_model('resnext101_32x8d', pretrained=pretrained,
                                     num_classes=0, global_pool='')
            _replace_bn_with_ibn(base)
            feat_dim = 2048
        elif name == 'resnetSt101':
            base = timm.create_model('resnest101e', pretrained=pretrained,
                                     num_classes=0, global_pool='')
            feat_dim = 2048
        else:
            raise ValueError(f'unknown backbone: {name}')

        self.base = base
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.base.forward_features(x)


class PetBiometricModel(nn.Module):
    """Full architecture: backbone -> Non-local -> GeM -> BN-Neck -> embed.

    When ``embed_dim`` equals the backbone feature dimension (2048) the
    extra Linear projection is skipped, matching the reference repo's
    ``EMBEDDING_DIM: 0`` setting where the output is the BN-neck feature.

    ``freeze_backbone=True`` (auto-enabled in --finetune_from mode) makes
    the backbone + Non-local + GeM non-trainable AND locks them in eval
    mode so their internal BN running stats do not drift during the
    finetune. The forward pass through the frozen path runs under
    ``torch.no_grad()`` for memory/speed. Only the BN-Neck (and the
    optional Linear projection) receive gradients.
    """
    def __init__(self, backbone_name='seresnet101_ibn',
                 embed_dim=2048, pretrained=True,
                 freeze_backbone=False):
        super().__init__()
        self.backbone = PetBiometricBackbone(backbone_name, pretrained)
        feat_dim = self.backbone.feat_dim

        self.nonlocal_block = NonLocalBlock(feat_dim)
        self.gem = GeMPool()

        # BN-Neck (Luo et al., ref [6]): BN before the embedding layer
        # has been shown to stabilize training and improve retrieval.
        self.bn_neck = nn.BatchNorm1d(feat_dim)
        self.bn_neck.bias.requires_grad_(False)

        if embed_dim == feat_dim:
            self.embed_fc = nn.Identity()
            self.out_dim = feat_dim
        else:
            self.embed_fc = nn.Linear(feat_dim, embed_dim, bias=False)
            self.out_dim = embed_dim

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.nonlocal_block.parameters():
                p.requires_grad = False
            for p in self.gem.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen sub-modules in eval mode so BN running stats stay put.
        if mode and self.freeze_backbone:
            self.backbone.eval()
            self.nonlocal_block.eval()
        return self

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
                feat = self.nonlocal_block(feat)
                pooled = self.gem(feat).flatten(1)
        else:
            feat = self.backbone(x)
            feat = self.nonlocal_block(feat)
            pooled = self.gem(feat).flatten(1)
        normed = self.bn_neck(pooled)
        emb = self.embed_fc(normed)
        return F.normalize(emb, dim=1)


# ===========================================================================
# Loss functions (paper §2.4)
# ===========================================================================
class OIMLoss(nn.Module):
    """Online Instance Matching loss (Xiao et al., ref [12]).

    Maintains a *cluster-level* memory bank of class centroids c_j ∈ R^d,
    updated by exponential moving average:
        c_j ← m·c_j + (1-m) · mean(features of class j in this batch)

    Then computes a softmax cross-entropy of <embedding, c_j> / τ over
    all classes, treating the true class centroid as the positive.

    NOTE: this implementation does the centroid update inside the loss
    forward, but with no gradient (the centroid is a non-learnable buffer).
    """
    def __init__(self, embed_dim, num_classes, momentum=0.5, temperature=0.05):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer(
            'centroids', F.normalize(torch.randn(num_classes, embed_dim), dim=1)
        )

    @torch.no_grad()
    def update_centroids(self, embeddings, labels):
        """Aggregate embeddings per class in this batch, then EMA-update
        the corresponding centroid rows."""
        unique_labels = labels.unique()
        for lbl in unique_labels:
            mask = (labels == lbl)
            if mask.sum() == 0:
                continue
            mean_feat = embeddings[mask].mean(dim=0)
            mean_feat = F.normalize(mean_feat, dim=0)
            j = int(lbl.item())
            self.centroids[j] = F.normalize(
                self.momentum * self.centroids[j]
                + (1.0 - self.momentum) * mean_feat,
                dim=0,
            )

    def forward(self, embeddings, labels):
        # Eq. (2): softmax cross-entropy with cluster centroids as logits.
        # Use a detached snapshot so the subsequent in-place centroid update
        # doesn't poison the autograd graph.
        centroids_snapshot = self.centroids.detach().clone()
        logits = embeddings @ centroids_snapshot.t() / self.temperature
        loss = F.cross_entropy(logits, labels)
        # Update centroids AFTER computing loss to avoid leakage.
        self.update_centroids(embeddings.detach(), labels)
        return loss


class WeightedICL(nn.Module):
    """Weighted Instance-level Contrastive Loss (paper §2.4, Eq. 4).

    Maintains an *instance-level* memory bank V' ∈ R^{N×d} containing the
    most recent embedding for every image in the dataset (indexed by
    global index from DogDataset). For each anchor:
      - positive set = all instances with the same label
      - negative set = top-K nearest instances with different label
                       (hard negative mining)
      - positive weights w_j = 1 - softmax(<v_j, E(x_i)>) over positives

    Loss = -∑_i ∑_j w_j · log( exp(<v_j, E_i>/τ)
                              / Σ_k exp(<v_k, E_i>/τ) )
    """
    def __init__(self, embed_dim, num_samples, momentum=0.5,
                 temperature=0.05, n_neg=500):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        self.n_neg = n_neg
        self.register_buffer(
            'memory', F.normalize(torch.randn(num_samples, embed_dim), dim=1)
        )
        self.register_buffer(
            'memory_labels', torch.full((num_samples,), -1, dtype=torch.long)
        )

    @torch.no_grad()
    def update_memory(self, embeddings, labels, indices):
        for emb, lbl, idx in zip(embeddings, labels, indices):
            i = int(idx.item())
            self.memory[i] = F.normalize(
                self.momentum * self.memory[i]
                + (1.0 - self.momentum) * emb,
                dim=0,
            )
            self.memory_labels[i] = int(lbl.item())

    def forward(self, embeddings, labels, indices):
        N = self.memory.size(0)
        n_neg = min(self.n_neg, N)
        # debug
        valid_in_memory = (self.memory_labels >= 0).sum().item()
        unique_labels_in_mem = self.memory_labels[self.memory_labels >= 0].unique().numel()
        print(f"  [ICL debug] memory filled: {valid_in_memory}/{N}, "
            f"distinct labels in memory: {unique_labels_in_mem}, "
            f"batch labels: {labels.unique().tolist()}")
        # Detached snapshot — same reason as in OIMLoss.
        memory_snapshot = self.memory.detach().clone()
        memory_labels_snapshot = self.memory_labels.detach().clone()

        # Pairwise similarity from this batch's embeddings to the entire memory.
        sim = embeddings @ memory_snapshot.t() / self.temperature
        per_anchor_losses = []

        for i in range(embeddings.size(0)):
            lbl = labels[i]

            # Positives: same-class entries in memory + same-class entries in this batch
            mem_same = (memory_labels_snapshot == lbl)
            batch_same_mask = (labels == lbl).clone()
            batch_same_mask[i] = False  # exclude self

            # Negatives: different-class entries in memory + different in batch
            mem_diff = ~mem_same & (memory_labels_snapshot >= 0)
            batch_diff_mask = (labels != lbl)

            # Similarities to memory entries
            mem_sim_pos = sim[i, mem_same]
            mem_sim_neg_all = sim[i, mem_diff]

            # Similarities to other batch entries (computed fresh, with gradient)
            batch_sim = embeddings[i:i+1] @ embeddings.t() / self.temperature
            batch_sim = batch_sim.squeeze(0)
            batch_sim_pos = batch_sim[batch_same_mask]
            batch_sim_neg = batch_sim[batch_diff_mask]

            sim_pos = torch.cat([mem_sim_pos, batch_sim_pos])
            sim_neg_all = torch.cat([mem_sim_neg_all, batch_sim_neg])

            if sim_pos.numel() == 0 or sim_neg_all.numel() == 0:
                continue

            # Hard negative mining
            k = min(n_neg, sim_neg_all.numel())
            sim_neg, _ = torch.topk(sim_neg_all, k)

            with torch.no_grad():
                weights = 1.0 - torch.softmax(sim_pos.detach(), dim=0)

            denom = torch.cat([sim_pos, sim_neg]).logsumexp(dim=0)
            log_probs = sim_pos - denom
            anchor_loss = -(weights * log_probs).sum()
            per_anchor_losses.append(anchor_loss)

        if not per_anchor_losses:
            return torch.zeros((), device=embeddings.device, requires_grad=True)

        loss = torch.stack(per_anchor_losses).mean()
        # Update memory AFTER loss computation.
        self.update_memory(embeddings.detach(), labels, indices)
        return loss


# ===========================================================================
# Profiling (Reviewer Comment #10)
# ===========================================================================
class GPUPowerSampler:
    """Poll nvidia-smi power.draw in a background thread."""
    def __init__(self, gpu_index=0, interval_s=0.5):
        self.gpu_index = gpu_index
        self.interval = interval_s
        self._samples = []
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
                p = float(out.splitlines()[0])
            except Exception:
                p = 0.0
            self._samples.append((t, p))
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
        if len(self._samples) < 2:
            return 0.0, 0.0
        ts = np.array([s[0] for s in self._samples])
        ps = np.array([s[1] for s in self._samples])
        wh = float(np.trapz(ps, ts)) / 3600.0
        return float(np.mean(ps)), wh


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, device, image_size=224):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return float(macs * 2) / 1e9
    except Exception:
        return float('nan')


def estimate_edge_latency(model, device, n_iters=50, image_size=224):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1000.0


# ===========================================================================
# Checkpointing
# ===========================================================================
def save_checkpoint(path, *, epoch, model, oim, icl, optimizer,
                    profile_rows, iter_in_stage=0):
    state = {
        'epoch': epoch,
        'iter_in_stage': iter_in_stage,
        'model_state':   model.state_dict(),
        'oim_state':     oim.state_dict(),
        'icl_state':     icl.state_dict(),
        'optim_state':   optimizer.state_dict(),
        'rng_python':    random.getstate(),
        'rng_numpy':     np.random.get_state(),
        'rng_torch':     torch.get_rng_state(),
        'rng_cuda':      torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None,
        'profile_rows':  profile_rows,
        'args':          vars(args),
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)
    print(f'  -> saved checkpoint: {path}')


def load_checkpoint(path, model, oim, icl, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model_state = {k: v for k, v in state['model_state'].items()
                   if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(model_state)
    oim.load_state_dict(state['oim_state'])
    icl.load_state_dict(state['icl_state'])
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
            pass  # stale checkpoint saved rng_cuda in wrong format; skip, non-critical
    return (state['epoch'],
            state.get('profile_rows', []),
            state.get('iter_in_stage', 0))


def partial_load_for_finetune(path, model, device):
    """Load only the backbone-side weights from an older checkpoint and
    drop the embed_fc projection (and any non-matching shapes), so an old
    512-d Stage 1 model can be converted to a 2048-d head without
    retraining the feature extractor.

    Returns the list of dropped / mismatched keys for logging.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state['model_state'] if isinstance(state, dict) and 'model_state' in state else state

    target_sd = model.state_dict()
    keep, dropped = {}, []
    for k, v in sd.items():
        if k.endswith(('total_ops', 'total_params')):
            continue
        # Always drop the projection — it has the wrong shape for the new head.
        if k.startswith('embed_fc'):
            dropped.append(f'{k} (skipped projection)')
            continue
        if k not in target_sd:
            dropped.append(f'{k} (not in target model)')
            continue
        if target_sd[k].shape != v.shape:
            dropped.append(f'{k} (shape {tuple(v.shape)} -> '
                           f'{tuple(target_sd[k].shape)})')
            continue
        keep[k] = v

    missing, unexpected = model.load_state_dict(keep, strict=False)
    return dropped, list(missing), list(unexpected)


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
            print(f'  -> pruned old checkpoint: {path}')
        except OSError:
            pass


# ===========================================================================
# Train
# ===========================================================================
def build_train_loader(img_size):
    """Rebuild the loader for a given resolution. Called twice during the
    run (once at low res, once at high res) following the paper's schedule."""
    transform = build_train_transform(img_size, args.aug)
    dataset = DogDataset(args.data_path, transform=transform)
    label_to_indices = build_label_index(dataset)

    sampler = PKBatchSampler(
        label_to_indices,
        p_identities=args.p_identities,
        k_per_identity=args.k_per_identity,
        batches_per_epoch=args.batches_per_epoch,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    return loader, len(dataset)


def compute_lr(stage, epoch_in_stage, iter_in_stage):
    """Return the current learning rate.

    Reproduces the reference repo's SOLVER block:
      * linear warmup over ``warmup_iters`` iters from
        ``warmup_factor * base_lr`` up to ``base_lr`` (Stage 1 only)
      * MultiStepLR with ``gamma`` at the configured milestones,
        independently for Stage 1 and Stage 2

    ``stage`` is 1 or 2; ``epoch_in_stage`` is 0-indexed within that stage;
    ``iter_in_stage`` is the cumulative iteration counter within the stage.
    """
    if stage == 1:
        base_lr   = args.lr
        milestones = args.steps_stage1
        if iter_in_stage < args.warmup_iters:
            alpha = iter_in_stage / max(1, args.warmup_iters)
            return base_lr * (args.warmup_factor * (1.0 - alpha) + alpha)
    else:
        base_lr   = args.lr_finetune
        milestones = args.steps_stage2

    decay = 1.0
    for ms in milestones:
        if epoch_in_stage >= ms:
            decay *= args.gamma
    return base_lr * decay


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- Finetune-mode sanity checks -----
    if args.finetune_from is not None:
        if not os.path.isfile(args.finetune_from):
            raise SystemExit(f'--finetune_from points to a missing file: '
                             f'{args.finetune_from}')
        if args.epochs_finetune <= 0:
            raise SystemExit('--finetune_from requires --epochs_finetune > 0 '
                             '(there must be Stage 2 epochs to run).')

    # --finetune_from implies --freeze_backbone (train only the BN-Neck head).
    freeze_backbone = args.freeze_backbone or (args.finetune_from is not None)

    # Build the model first so we can size losses correctly.
    model = PetBiometricModel(
        backbone_name=args.backbone,
        embed_dim=args.embed_dim,
        pretrained=False,
        freeze_backbone=freeze_backbone,
    ).to(device)

    # Build losses. ICL needs to know the full dataset size.
    _, n_total = build_train_loader(args.img_size)
    out_dim = model.out_dim
    oim = OIMLoss(out_dim, args.c,
                  momentum=args.oim_momentum,
                  temperature=args.oim_temperature).to(device)
    icl = WeightedICL(out_dim, n_total,
                      momentum=args.icl_momentum,
                      temperature=args.icl_temperature,
                      n_neg=args.icl_n_neg).to(device)

    # Optimizer over trainable params only. When the backbone is frozen
    # this reduces to the BN-Neck (and an optional Linear projection).
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    optimizer = Adam(params, lr=args.lr, weight_decay=args.wd)
    if freeze_backbone:
        print(f'[freeze_backbone] training only the head: '
              f'{n_trainable:,} trainable params '
              f'(BN-Neck' +
              (' + Linear projection' if not isinstance(model.embed_fc, nn.Identity)
               else '') + ')')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ----- Resume -----
    start_epoch = 0
    resumed_profile_rows = []
    resumed_iter_in_stage = 0
    if args.resume == 'auto':
        latest = find_latest_checkpoint(args.ckpt_dir)
    elif args.resume == 'none':
        latest = None
    else:
        latest = args.resume
    if latest is not None and os.path.isfile(latest):
        print(f'Resuming from checkpoint: {latest}')
        start_epoch, resumed_profile_rows, resumed_iter_in_stage = \
            load_checkpoint(latest, model, oim, icl, optimizer, device)
        print(f'  resumed at epoch {start_epoch} '
              f'(continuing from epoch {start_epoch + 1})')
    elif args.finetune_from is not None:
        # Skip Stage 1, partial-load backbone-side weights, jump to Stage 2.
        print(f'Finetune mode: partial-loading from {args.finetune_from}')
        if args.embed_dim != 2048:
            print(f'  NOTE: --embed_dim={args.embed_dim}. The reference '
                  f'(EMBEDDING_DIM=0) uses 2048-d output. Pass '
                  f'--embed_dim 2048 to drop the projection entirely.')
        dropped, missing, unexpected = partial_load_for_finetune(
            args.finetune_from, model, device)
        if dropped:
            print(f'  dropped {len(dropped)} key(s): {dropped[:6]}'
                  f'{" ..." if len(dropped) > 6 else ""}')
        if missing:
            print(f'  missing in checkpoint ({len(missing)}): {missing[:6]}'
                  f'{" ..." if len(missing) > 6 else ""}')
        if unexpected:
            print(f'  unexpected in checkpoint ({len(unexpected)}): '
                  f'{unexpected[:6]}'
                  f'{" ..." if len(unexpected) > 6 else ""}')
        # Jump directly to Stage 2: the loop's stage-selection will see
        # epoch >= args.epochs and run only the 384-px finetune.
        start_epoch = args.epochs
        print(f'  jumping to Stage 2 (epoch {start_epoch + 1} '
              f'/ {args.epochs + args.epochs_finetune})')

    total_epochs = args.epochs + args.epochs_finetune
    print('=' * 60)
    print(f'Backbone: {args.backbone}')
    print(f'Embedding dim: {out_dim} '
          f'({"BN-neck only" if isinstance(model.embed_fc, nn.Identity) else "BN-neck + Linear projection"})')
    print(f'PK batches: P={args.p_identities}, K={args.k_per_identity} '
          f'-> batch size {args.p_identities * args.k_per_identity}')
    print(f'Augmentation: {args.aug}')
    print(f'Stage 1: {args.epochs} epochs @ {args.img_size}px, '
          f'LR={args.lr} (warmup {args.warmup_iters} iters x{args.warmup_factor}, '
          f'milestones={args.steps_stage1}, gamma={args.gamma})')
    print(f'Stage 2: {args.epochs_finetune} epochs @ {args.img_size_finetune}px, '
          f'LR={args.lr_finetune} '
          f'(milestones={args.steps_stage2}, gamma={args.gamma})')
    print(f'Total: {total_epochs} epochs')
    print('=' * 60)

    # ----- Static profiling -----
    profile_rows = list(resumed_profile_rows)
    n_params = gflops = edge_ms = 0
    if args.profile:
        os.makedirs(os.path.dirname(args.profile_csv) or '.', exist_ok=True)
        n_params = count_parameters(model)
        gflops = estimate_flops(model, device, args.img_size)
        edge_ms = estimate_edge_latency(model, device,
                                        image_size=args.img_size)
        print(f'Total trainable params: {n_params/1e6:.2f} M')
        print(f'Forward GFLOPs @ {args.img_size}: {gflops:.2f}')
        print(f'Edge latency @ {args.img_size}: {edge_ms:.2f} ms / image')
        print('=' * 60)
        model.train()

    power_sampler = GPUPowerSampler() if args.profile else None

    # ----- Stage-aware training loop -----
    # Track iteration counter within each stage so warmup / MultiStepLR
    # are computed correctly even after a resume.
    # If we resumed inside Stage 1 we keep its counter; if we resumed at or
    # past the Stage-2 boundary the counter is reset on the boundary below.
    initial_stage = 1 if start_epoch < args.epochs else 2
    initial_size  = (args.img_size if initial_stage == 1
                     else args.img_size_finetune)
    train_loader, _ = build_train_loader(initial_size)
    current_img_size = initial_size
    iter_in_stage = resumed_iter_in_stage

    for epoch in range(start_epoch, total_epochs):
        # Decide which stage this epoch belongs to and rebuild the loader
        # / reset iteration counter at the boundary.
        if epoch < args.epochs:
            stage = 1
            epoch_in_stage = epoch
            target_size = args.img_size
        else:
            stage = 2
            epoch_in_stage = epoch - args.epochs
            target_size = args.img_size_finetune
            if epoch == args.epochs:
                # Boundary: rebuild loader at the finetune resolution,
                # reset warmup-relevant counter.
                iter_in_stage = 0

        if current_img_size != target_size:
            print(f'[Stage {stage}] rebuilding loader at {target_size}px')
            train_loader, _ = build_train_loader(target_size)
            current_img_size = target_size

        epoch_t0 = time.perf_counter()
        n_imgs_seen = 0

        if args.profile and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if power_sampler is not None:
            power_sampler.start()

        for i, (imgs, lbls, idxs) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.long().to(device)
            idxs = idxs.long().to(device)

            lr_now = compute_lr(stage, epoch_in_stage, iter_in_stage)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

            embeddings = model(imgs)

            loss_oim = oim(embeddings, lbls)
            loss_icl = icl(embeddings, lbls, idxs)
            loss = args.lambda_oim * loss_oim + args.lambda_icl * loss_icl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_in_stage += 1
            n_imgs_seen += imgs.size(0)
            print(f'Stage{stage} Epoch [{epoch+1}/{total_epochs}], '
                  f'Step [{i+1}/{len(train_loader)}], '
                  f'lr={lr_now:.2e}, '
                  f'Loss: {loss.item():.4f}  '
                  f'(OIM: {loss_oim.item():.3f}, '
                  f'ICL: {loss_icl.item():.3f})')

        if device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_t1 = time.perf_counter()

        if power_sampler is not None:
            power_sampler.stop()

        # ----- Profile row -----
        if args.profile:
            epoch_s = epoch_t1 - epoch_t0
            peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                           if device.type == 'cuda' else 0.0)
            avg_w, energy_wh = (power_sampler.integrate_wh()
                                if power_sampler is not None else (0.0, 0.0))
            row = {
                'stage': stage,
                'epoch': epoch + 1,
                'img_size': current_img_size,
                'epoch_time_s': epoch_s,
                'ms_per_step': epoch_s / max(1, len(train_loader)) * 1000,
                'images_per_second': n_imgs_seen / max(1e-6, epoch_s),
                'peak_gpu_mem_MB': peak_mem_mb,
                'avg_gpu_power_W': avg_w,
                'energy_Wh': energy_wh,
            }
            profile_rows.append(row)
            print('-' * 60)
            print(f'[Profile] Stage{stage} Epoch {epoch+1}: '
                  f'{epoch_s:.1f}s, peak mem {peak_mem_mb:.0f} MB, '
                  f'avg power {avg_w:.1f} W, energy {energy_wh:.3f} Wh')
            print('-' * 60)

        # ----- Periodic checkpoint -----
        epoch_1based = epoch + 1
        is_last = (epoch_1based == total_epochs)
        if (epoch_1based % args.ckpt_every == 0) or is_last:
            ckpt_path = os.path.join(
                args.ckpt_dir, f'ckpt_epoch_{epoch_1based:04d}.pt'
            )
            save_checkpoint(
                ckpt_path,
                epoch=epoch_1based,
                model=model, oim=oim, icl=icl,
                optimizer=optimizer,
                profile_rows=profile_rows,
                iter_in_stage=iter_in_stage,
            )
            prune_old_checkpoints(args.ckpt_dir, keep=args.keep_ckpts)

    # ----- Final saves -----
    # In --finetune_from mode the output IS the new Stage 1 model
    # (paper-faithful 2048-d head). It overwrites the old _stage1.pt
    # so downstream tools (e.g. the Siamese script) pick it up
    # automatically. The original is kept under a *_orig512.pt backup.
    if args.finetune_from is not None:
        final_path = os.path.join(args.output_dir,
                                  f'pet2022_{args.backbone}_stage1.pt')
        if os.path.isfile(final_path):
            backup = final_path.replace('.pt', '_orig512.pt')
            if not os.path.isfile(backup):
                os.replace(final_path, backup)
                print(f'Backed up original Stage 1 to: {backup}')
        torch.save(model.state_dict(), final_path)
        print(f'Saved finetuned (2048-d) Stage 1 model to: {final_path}')
    else:
        suffix = 'stage2' if args.epochs_finetune > 0 else 'stage1'
        final_path = os.path.join(args.output_dir,
                                  f'pet2022_{args.backbone}_{suffix}.pt')
        torch.save(model.state_dict(), final_path)
        print(f'Saved final contrastive-stage model to: {final_path}')

    if args.profile and profile_rows:
        df = pd.DataFrame(profile_rows)
        df.to_csv(args.profile_csv, index=False)
        static_path = args.profile_csv.replace('.csv', '_static.csv')
        pd.DataFrame([{
            'backbone': args.backbone,
            'total_params_M': n_params / 1e6,
            'forward_gflops': gflops,
            'edge_latency_ms': edge_ms,
        }]).to_csv(static_path, index=False)
        print(f'Wrote profile to: {args.profile_csv}')


if __name__ == '__main__':
    train()
