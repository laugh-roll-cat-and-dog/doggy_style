"""
Gallery-based open/closed-set evaluation using only Stage 1 (contrastive)
backbones — no Siamese head. The pair score is plain cosine similarity:

    score = <test_feat, centroid>

Across models (paper §2.6.2 score ensemble):
    final = mean_over_models(cosine)

Use this to measure what the Stage 1 contrastive features alone can do,
before adding the Siamese verification branch from Stage 2.

Defaults target the **current** Stage 1 layout produced by
``pet_biometric_2022.py``:
  - 2048-d output (EMBEDDING_DIM:0 — no embed_fc projection)
  - finetune resolution 384 px

If you want to evaluate the older 512-d checkpoints (e.g. the
``*_orig512.pt`` backups), pass ``--embed_dim 512``.

Backbone names follow the current ``--backbone`` choices:
  seresnet101_ibn / resnetXt101_ibn / resnetSt101

Layout (matches evaluate_convnext_2.py):
    crop_copy/<class_id>/gallery/   first N images per class
    crop_copy/<class_id>/test/      known-ID queries
    crop_copy/unknown/              unknown-ID queries (label = -1)

Example:
    uv run ensemble_eval_stage1.py \
      --model seresnet101_ibn:model/pet2022/pet2022_seresnet101_ibn_stage1.pt \
      --model resnetXt101_ibn:model/pet2022/pet2022_resnetXt101_ibn_stage1.pt \
      --model resnetSt101:model/pet2022/pet2022_resnetSt101_stage1.pt \
      --include_unknown
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt

from pet_biometric_2022 import PetBiometricModel, build_val_transform


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='crop_copy')
parser.add_argument('--label_min', type=int, default=17)
parser.add_argument('--label_max', type=int, default=70)
parser.add_argument('--gallery_per_id', type=int, default=4)
parser.add_argument('--img_size', type=int, default=384,
                    help='Query image size. Default matches the finetune '
                         'resolution of the current Stage 1 model (384).')
parser.add_argument('--gallery_img_size', type=int, default=384,
                    help='Gallery image size. Default 384 to match the '
                         'current Stage 1 finetune resolution. Pass 224 '
                         'if you really want low-res gallery.')
parser.add_argument('--model', action='append', required=True,
                    help='backbone_name:path/to/stage1.pt  (repeat for '
                         'ensemble). Backbone names: seresnet101_ibn / '
                         'resnetXt101_ibn / resnetSt101 (current --backbone '
                         'choices in pet_biometric_2022.py).')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--far_target', type=float, default=0.01)
parser.add_argument('--include_unknown', action='store_true')
parser.add_argument('--output_prefix', type=str, default='pet2022_stage1_ensemble')
parser.add_argument('--embed_dim', type=int, default=2048,
                    help='Embedding dim. 2048 = current EMBEDDING_DIM:0 '
                         'layout (no embed_fc projection). Use 512 only '
                         'when evaluating an old un-finetuned checkpoint '
                         '(e.g. the *_orig512.pt backups). MUST match the '
                         'dimension all listed --model checkpoints were '
                         'trained at.')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GalleryDogDataset(Dataset):
    def __init__(self, root_dir, split, transform,
                 label_min=17, label_max=70, gallery_per_id=4):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []

        if split == 'unknown':
            unknown_dir = os.path.join(root_dir, 'unknown')
            if not os.path.isdir(unknown_dir):
                return
            for root, _, files in os.walk(unknown_dir):
                for fname in natsorted(files):
                    fpath = os.path.join(root, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(
                            ('.png', '.jpg', '.jpeg')):
                        self.samples.append((fpath, -1))
            return

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                label = int(class_name)
            except ValueError:
                continue
            if not (label_min <= label < label_max):
                continue
            for root, _, files in os.walk(class_dir):
                if os.path.basename(root) != split:
                    continue
                for i, fname in enumerate(natsorted(files)):
                    fpath = os.path.join(root, fname)
                    if not os.path.isfile(fpath):
                        continue
                    if split == 'gallery' and i >= gallery_per_id:
                        continue
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_path


# ---------------------------------------------------------------------------
# Feature extraction (Stage 1 only — no Siamese head)
# ---------------------------------------------------------------------------
def parse_model_specs(specs):
    out = []
    for s in specs:
        if ':' not in s:
            raise ValueError(f"--model must be 'backbone:path', got: {s}")
        bb, path = s.split(':', 1)
        out.append((bb.strip(), path.strip()))
    return out


def load_model(backbone, path, device):
    model = PetBiometricModel(backbone_name=backbone,
                              embed_dim=args.embed_dim,
                              pretrained=False).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state['model_state'] if 'model_state' in state else state
    # Strip thop-style profiling buffers if present.
    sd = {k: v for k, v in sd.items()
          if 'total_ops' not in k and 'total_params' not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Most common cause: --embed_dim 2048 vs a 512-d checkpoint
        # (then embed_fc.weight is unexpected) or the reverse (missing).
        msg = (f'state_dict mismatch loading {path} '
               f'(backbone={backbone}, --embed_dim={args.embed_dim}):\n'
               f'  missing ({len(missing)}): {list(missing)[:5]}'
               f'{" ..." if len(missing) > 5 else ""}\n'
               f'  unexpected ({len(unexpected)}): {list(unexpected)[:5]}'
               f'{" ..." if len(unexpected) > 5 else ""}')
        # Only fatal if it's not just embed_fc differences we can ignore.
        only_embed = (
            all(m.startswith('embed_fc') for m in missing)
            and all(u.startswith('embed_fc') for u in unexpected)
        )
        if only_embed:
            print(f'WARNING: {msg}')
            print('  Continuing with un-initialised embed_fc — pass '
                  '--embed_dim that matches the checkpoint to fix.')
        else:
            raise RuntimeError(msg)
    model.eval()
    return model


def extract_features(model, loader, device):
    feats, labels, paths = [], [], []
    with torch.no_grad():
        for imgs, lbls, p in loader:
            imgs = imgs.to(device)
            emb = model(imgs)               # already L2-normalised
            feats.append(emb.cpu())
            labels.extend(lbls.tolist() if isinstance(lbls, torch.Tensor)
                          else list(lbls))
            paths.extend(list(p))
    return torch.cat(feats, dim=0), labels, paths


def build_centroids(g_feats, g_labels):
    by_label = {}
    for emb, lbl in zip(g_feats, g_labels):
        by_label.setdefault(int(lbl), []).append(emb)
    ordered_labels = sorted(by_label.keys())
    centroids = torch.stack([
        F.normalize(torch.stack(by_label[l]).mean(dim=0), dim=0)
        for l in ordered_labels
    ])
    return centroids, ordered_labels


def cosine_scores(q_feats, centroids, device):
    return (q_feats.to(device) @ centroids.to(device).t()).cpu()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_eer_threshold(sim_matrix, test_labels, centroid_labels,
                          output_prefix, far_target):
    label_mask = (test_labels.unsqueeze(1) == centroid_labels.unsqueeze(0))

    pos = sim_matrix.clone(); pos[~label_mask] = -float('inf')
    neg = sim_matrix.clone(); neg[label_mask]  = -float('inf')
    pos_scores = pos.max(dim=1).values.cpu().numpy()
    neg_scores = neg.max(dim=1).values.cpu().numpy()
    pos_scores = pos_scores[np.isfinite(pos_scores)]
    neg_scores = neg_scores[np.isfinite(neg_scores)]

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)),
                             np.zeros(len(neg_scores))])
    fpr, tpr, thr = roc_curve(labels, scores)
    far, frr = fpr, 1 - tpr
    order = np.argsort(thr)
    thr, far, frr = thr[order], far[order], frr[order]

    diff = far - frr
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        i = int(np.argmin(np.abs(diff)))
        eer_thr = thr[i]
        eer = (far[i] + frr[i]) / 2
    else:
        i = idx[0]
        x0, x1 = thr[i], thr[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        eer_thr = x0 - y0 * (x1 - x0) / (y1 - y0)
        eer = float(np.interp(eer_thr, thr, far))

    far_idx = np.where(fpr <= far_target)[0]
    far_threshold = float(thr[far_idx[-1]]) if len(far_idx) else float('nan')

    os.makedirs('result_csv', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(thr, far, label='FAR', linewidth=2)
    plt.plot(thr, frr, label='FRR', linewidth=2)
    plt.scatter(eer_thr, eer, color='red', zorder=5, label=f'EER = {eer:.4f}')
    plt.axvline(eer_thr, color='red', linestyle='--', alpha=0.7)
    plt.yscale('log'); plt.ylim(1e-2, 1)
    plt.xlabel('Score threshold'); plt.ylabel('Error rate')
    plt.title('FAR–FRR intersection (EER)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'result_csv/{output_prefix}_eer_intersection.png')
    plt.close()

    print('=' * 30)
    print(f'EER:        {eer:.4f}')
    print(f'EER thr:    {eer_thr:.4f}')
    print(f'FAR@{far_target} thr: {far_threshold:.4f}')
    print('=' * 30)
    return float(eer_thr), far_threshold, float(eer)


def evaluate_closed_set(sim_matrix, test_labels, centroid_labels,
                        test_paths, output_prefix, eer_thr, far_thr,
                        top_k):
    topk_scores, topk_idx = torch.topk(sim_matrix, k=top_k, dim=1)
    pred_labels = centroid_labels[topk_idx]
    top1_scores = topk_scores[:, 0]

    accept_eer = top1_scores >= eer_thr
    accept_far = top1_scores >= far_thr
    pred_eer = pred_labels[:, 0].clone(); pred_eer[~accept_eer] = -1
    pred_far = pred_labels[:, 0].clone(); pred_far[~accept_far] = -1

    label_mask = (test_labels.unsqueeze(1) == centroid_labels.unsqueeze(0))
    masked = sim_matrix.clone(); masked[~label_mask] = -float('inf')
    true_class_scores = masked.max(dim=1).values

    y_binary = (pred_labels[:, 0] == test_labels).int().cpu().numpy()
    y_scores = true_class_scores.cpu().numpy()
    valid = y_scores >= 0
    fpr, tpr, _ = roc_curve(y_binary[valid], y_scores[valid])
    roc_auc = auc(fpr, tpr)

    correct = pred_labels.eq(test_labels.view(-1, 1).expand_as(pred_labels))
    accs = [correct[:, :k+1].any(dim=1).float().mean().item() * 100
            for k in range(top_k)]
    print(f'Closed-set ROC AUC: {roc_auc:.4f}')
    for k, a in enumerate(accs, 1):
        print(f'Top-{k} Accuracy: {a:.2f}%')

    y_true = test_labels.cpu().numpy()
    y_pred = pred_labels[:, 0].cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    summary = {'output_prefix': output_prefix, 'closed_set_auc': roc_auc,
               'precision': precision, 'recall': recall, 'f1_score': f1,
               'num_test_samples': len(y_true)}
    for k, a in enumerate(accs, 1):
        summary[f'top{k}_accuracy'] = a
    pd.DataFrame([summary]).to_csv(
        f'result_csv/{output_prefix}_metrics.csv', index=False
    )
    print(f'Saved metrics summary to: result_csv/{output_prefix}_metrics.csv')

    rows = []
    t_scores_np = true_class_scores.cpu().numpy()
    for i in range(len(y_true)):
        ts = t_scores_np[i]
        if ts == -float('inf'):
            ts = -1.0
        rows.append({
            'filename': test_paths[i],
            'true_id': int(y_true[i]),
            'pred_top1': int(pred_labels[i, 0].item()),
            'is_correct_closed': bool(y_true[i] == pred_labels[i, 0].item()),
            'score_top1': float(topk_scores[i, 0].item()),
            'pred_eer': int(pred_eer[i].item()),
            'pred_far': int(pred_far[i].item()),
            'accept_eer': bool(accept_eer[i].item()),
            'accept_far': bool(accept_far[i].item()),
            'score_true_class': float(ts),
        })
    df = pd.DataFrame(rows)
    df['confidence_gap'] = df['score_top1'] - df['score_true_class']
    df.to_csv(f'result_csv/{output_prefix}_predictions.csv', index=False)
    return df


def evaluate_open_set(sim_open, open_labels, open_paths,
                      centroid_labels, threshold, output_prefix):
    top1_scores, top1_idx = sim_open.max(dim=1)
    pred = centroid_labels[top1_idx]
    accept = top1_scores >= threshold
    pred_open = pred.clone(); pred_open[~accept] = -1

    rows = []
    for i in range(len(open_labels)):
        true_id = int(open_labels[i].item())
        pred_id = int(pred_open[i].item())
        sc = float(top1_scores[i].item())
        is_unknown = (true_id == -1)
        is_correct = ((not accept[i]) if is_unknown
                      else (accept[i].item() and pred_id == true_id))
        rows.append({
            'filename': open_paths[i], 'true_id': true_id,
            'pred_id': pred_id, 'score': sc,
            'accepted': bool(accept[i].item()),
            'is_unknown': is_unknown, 'correct_open': bool(is_correct),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f'result_csv/{output_prefix}_open_set_samples.csv', index=False)

    known = df[df.true_id != -1]
    unknown = df[df.true_id == -1]
    n_k, n_u = len(known), len(unknown)
    known_correct = ((known.accepted) & (known.pred_id == known.true_id)).sum()
    known_frej    = (~known.accepted).sum()
    known_misid   = ((known.accepted) & (known.pred_id != known.true_id)).sum()
    known_acc = known_correct / n_k * 100 if n_k else np.nan
    known_frr = known_frej   / n_k * 100 if n_k else np.nan
    known_mir = known_misid  / n_k * 100 if n_k else np.nan
    far = unknown.accepted.mean() * 100 if n_u else np.nan
    trr = 100 - far if not np.isnan(far) else np.nan

    y_true = (df.true_id != -1).astype(int)
    y_score = df.score.values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('FAR'); plt.ylabel('TAR'); plt.title('Open-Set ROC')
    plt.legend(); plt.grid(True)
    plt.savefig(f'result_csv/{output_prefix}_open_set_roc.png'); plt.close()

    summary = pd.DataFrame([{
        'threshold_type': 'EER (FAR=FRR)', 'threshold': threshold,
        'known_id_accuracy_%': known_acc,
        'known_false_reject_rate_%': known_frr,
        'known_misidentification_rate_%': known_mir,
        'false_accept_rate_%': far, 'true_reject_rate_%': trr,
        'open_set_roc_auc': roc_auc,
    }])
    summary.to_csv(f'result_csv/{output_prefix}_open_set_summary.csv',
                   index=False)

    print('=' * 40)
    print('OPEN-SET (EER THRESHOLD)')
    print(f'Threshold              : {threshold:.4f}')
    print(f'Known-ID Accuracy      : {known_acc:.2f}%')
    print(f'Known False Reject     : {known_frr:.2f}%')
    print(f'Known Mis-ID Rate      : {known_mir:.2f}%')
    print(f'False Accept (unknown) : {far:.2f}%')
    print(f'True Reject (unknown)  : {trr:.2f}%')
    print(f'Open-set ROC AUC       : {roc_auc:.4f}')
    print('=' * 40)
    return summary, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gallery_tf = build_val_transform(args.gallery_img_size)
    query_tf   = build_val_transform(args.img_size)

    gallery_ds = GalleryDogDataset(
        args.data_path, 'gallery', gallery_tf,
        label_min=args.label_min, label_max=args.label_max,
        gallery_per_id=args.gallery_per_id,
    )
    test_ds = GalleryDogDataset(
        args.data_path, 'test', query_tf,
        label_min=args.label_min, label_max=args.label_max,
    )
    if len(gallery_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            'Empty gallery or test split — check directory layout: '
            '<root>/<class_id>/gallery/* and <root>/<class_id>/test/*'
        )
    print(f'Gallery: {len(gallery_ds)} imgs  |  Test: {len(test_ds)} imgs')

    unknown_ds = None
    if args.include_unknown:
        unknown_ds = GalleryDogDataset(
            args.data_path, 'unknown', query_tf,
            label_min=args.label_min, label_max=args.label_max,
        )
        print(f'Unknown: {len(unknown_ds)} imgs')

    g_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False)
    t_loader = DataLoader(test_ds,    batch_size=args.batch_size, shuffle=False)
    u_loader = (DataLoader(unknown_ds, batch_size=args.batch_size,
                           shuffle=False) if unknown_ds is not None else None)

    specs = parse_model_specs(args.model)
    print(f'Stage-1 score ensemble of {len(specs)} model(s):')
    for bb, p in specs:
        print(f'  {bb}: {p}')

    summed_test_scores = None
    summed_unknown_scores = None
    centroid_labels = None
    test_labels = None
    test_paths = None
    unknown_labels = None
    unknown_paths = None

    for bb, path in specs:
        print(f'\n--- {bb} ---')
        model = load_model(bb, path, device)

        g_feats, g_lbls, _   = extract_features(model, g_loader, device)
        t_feats, t_lbls, t_p = extract_features(model, t_loader, device)
        centroids, c_lbls    = build_centroids(g_feats, g_lbls)

        score_test = cosine_scores(t_feats, centroids, device)
        if summed_test_scores is None:
            summed_test_scores = score_test.clone()
            centroid_labels = torch.tensor(c_lbls)
            test_labels = torch.tensor(t_lbls)
            test_paths = t_p
        else:
            assert c_lbls == centroid_labels.tolist(), \
                'centroid label order changed between models'
            summed_test_scores += score_test

        if u_loader is not None:
            u_feats, u_lbls, u_p = extract_features(model, u_loader, device)
            score_unknown = cosine_scores(u_feats, centroids, device)
            if summed_unknown_scores is None:
                summed_unknown_scores = score_unknown.clone()
                unknown_labels = torch.tensor(u_lbls)
                unknown_paths = u_p
            else:
                summed_unknown_scores += score_unknown

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    final_test = summed_test_scores / len(specs)

    eer_thr, far_thr, eer = compute_eer_threshold(
        final_test, test_labels, centroid_labels,
        args.output_prefix, args.far_target
    )
    evaluate_closed_set(
        final_test, test_labels, centroid_labels, test_paths,
        args.output_prefix, eer_thr, far_thr, args.top_k
    )

    if summed_unknown_scores is not None:
        final_unknown = summed_unknown_scores / len(specs)
        open_scores = torch.cat([final_test, final_unknown], dim=0)
        open_labels = torch.cat([test_labels, unknown_labels], dim=0)
        open_paths  = list(test_paths) + list(unknown_paths)
        evaluate_open_set(
            open_scores, open_labels, open_paths,
            centroid_labels, eer_thr, args.output_prefix
        )


if __name__ == '__main__':
    main()
