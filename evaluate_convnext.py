import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import math
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_curve, auc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os
import cv2
from natsort import natsorted

from network.network import Network_Resnet, Network_ConvNext

from loss.arcface import ArcFace
from loss.softTriple import SoftTriple

RESOLUTION_LEVELS = {
    "R0": 1.00,
    "R1": 0.75,
    "R2": 0.50,
    "R3": 0.33,
    "R4": 0.25,
    "R5": 0.15,
}

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--attention', default='sb', choices=['s', 'b', 'sb', 'n'], type=str, metavar="attention",
                    help='attention module')
parser.add_argument('-b', '--backbone', choices=['dino', 'v2', 'resnet'], type=str, metavar="backbone",
                    help='backbone')
parser.add_argument('-m', '--model', type=str, metavar="model",
                    help=' model filename')
parser.add_argument('-r', '--resolution', choices=RESOLUTION_LEVELS.keys(), type=str, metavar="resolution",
                    help='input resolution')
parser.add_argument('-o', '--output', type=str, metavar="output",
                    help='output csv filename')
parser.add_argument('-d', '--dataset', choices=['face', 'nose', 'nose_old'], type=str, metavar="dataset",
                    help='what dataset to use')
args = parser.parse_args()

## Resolution degrade
class ResolutionDegradation:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        if self.scale == 1.0:
            return img

        img_np = np.array(img)  # HWC, RGB
        h, w = img_np.shape[:2]

        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))

        # Downscale (sensor simulation)
        img_down = cv2.resize(
            img_np, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        # Upscale back (ISP reconstruction)
        img_up = cv2.resize(
            img_down, (w, h), interpolation=cv2.INTER_CUBIC
        )

        return Image.fromarray(img_up)
resolution_scale = RESOLUTION_LEVELS[args.resolution]

## Data
gallery_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    ResolutionDegradation(scale=resolution_scale),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if args.dataset == 'face':
    class DogDataset(Dataset):
        def __init__(self,csv_file, transform=None):
            self.df = pd.read_csv(csv_file)
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, index):
            img_path = self.df.loc[index,'filepath']
            label = self.df.loc[index,'label']
            img_path_normalized = img_path.replace('\\', '/')
            image = Image.open(img_path_normalized).convert('RGB')

            if self.transform:
                image = self.transform(image)
            
            return image, label
    val_dataset = DogDataset(csv_file='test_split_mixed_sorted.csv', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    gallery_dataset = DogDataset(csv_file='train_split_mixed_sorted.csv', transform=val_transforms)
    gallery_loader = DataLoader(gallery_dataset, batch_size=16,shuffle=False)
else:
    class DogDataset(Dataset):
        def __init__(self, root_dir, split="train", transform=None, class_num=45):
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
                    if label > class_num - 1:
                        continue
                except:
                    continue

                for root, dirs, files in os.walk(class_dir):
                    if os.path.basename(root) != split:
                        continue

                    for i, fname in enumerate(natsorted(files)):
                        fpath = os.path.join(root, fname)
                        if os.path.isfile(fpath):
                            if split == 'train' and i >= 4:
                                continue
                            if label > 44:
                                label = -1
                            self.samples.append((fpath, label))


            if len(self.samples) == 0:
                raise ValueError(f"No images found for split='{split}' in {root_dir}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            img_path, label = self.samples[index]
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label, img_path
        
    val_dataset = DogDataset('crop', 'test', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    val_dataset_unknown = DogDataset('crop', 'test', transform=val_transforms, class_num=50)
    val_loader_unknown = DataLoader(val_dataset_unknown, batch_size=16, shuffle=False)

    gallery_dataset = DogDataset('crop', transform=gallery_transforms)
    gallery_loader = DataLoader(gallery_dataset, batch_size=16,shuffle=False)

def evaluate_embedding_metrics(test_embeddings, gallery_embeddings, output_prefix, top_k=5):
    gal_feats = []
    gal_labels = []
    for label, emb_list in gallery_embeddings.items():
        for emb in emb_list:
            gal_feats.append(emb)
            gal_labels.append(label)

    gal_feats = torch.stack(gal_feats).to(device)
    gal_feats = F.normalize(gal_feats, p=2, dim=1)
    gal_labels = torch.tensor(gal_labels).to(device)

    test_feats = []
    test_labels = []
    test_paths = []
    for label, emb_list in test_embeddings.items():
        for emb, path in emb_list:
            test_feats.append(emb)
            test_labels.append(label)
            test_paths.append(path)

    test_feats = torch.stack(test_feats).to(device)
    test_feats = F.normalize(test_feats, p=2, dim=1)
    test_labels_true = torch.tensor(test_labels).to(device)

    # 1. Compute Similarity Matrix
    sim_matrix = torch.matmul(test_feats, gal_feats.T)

    eer_thr, far_thr = compute_identification_threshold(
        sim_matrix,
        test_labels_true,
        gal_labels,
        output_prefix,
        target_far=0.01
    )

    # 2. Get Top-K Predictions
    topk_scores, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)
    pred_labels = gal_labels[topk_indices]

    top1_scores = topk_scores[:, 0]

    # Accept / Reject decisions
    accept_eer = top1_scores >= eer_thr
    accept_far = top1_scores >= far_thr

    # Threshold-aware predictions
    # -1 means "Unknown / Not in database"
    pred_eer = pred_labels[:, 0].clone()
    pred_far = pred_labels[:, 0].clone()

    pred_eer[~accept_eer] = -1
    pred_far[~accept_far] = -1

    # --- NEW BLOCK: Calculate Max Score for the True Class ---
    # Create a mask where (Test_i, Gal_j) is True if they have the same label
    # Shape: [num_test, num_gallery]
    label_mask = test_labels_true.unsqueeze(1) == gal_labels.unsqueeze(0)

    # Clone matrix to mask out wrong classes without affecting original sim_matrix
    masked_sim = sim_matrix.clone()
    
    # Set scores of non-matching classes to -infinity so they are ignored by max()
    masked_sim[~label_mask] = -float('inf')

    # Get the max similarity score for the true label for each test image
    true_class_scores = masked_sim.max(dim=1)[0]
    # ---------------------------------------------------------

    y_binary = (pred_labels[:, 0] == test_labels_true).int().cpu().numpy()
    y_scores = true_class_scores.cpu().numpy()

    # Remove invalid entries (if any)
    valid_mask = y_scores >= 0
    y_binary = y_binary[valid_mask]
    y_scores = y_scores[valid_mask]

    fpr, tpr, roc_thr = roc_curve(y_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"Closed-set ROC AUC: {roc_auc:.4f}")
    
    correct = pred_labels.eq(test_labels_true.view(-1, 1).expand_as(pred_labels))

    top1_acc = correct[:, 0].float().mean().item() * 100
    top2_acc = correct[:, :2].any(dim=1).float().mean().item() * 100
    top3_acc = correct[:, :3].any(dim=1).float().mean().item() * 100
    top4_acc = correct[:, :4].any(dim=1).float().mean().item() * 100
    top5_acc = correct[:, :5].any(dim=1).float().mean().item() * 100

    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-2 Accuracy: {top2_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"Top-4 Accuracy: {top4_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    y_true = test_labels_true.cpu().numpy()
    y_pred = pred_labels[:, 0].cpu().numpy()
    
    # Convert true scores to CPU for saving
    y_true_scores = true_class_scores.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    print("\n--- Detailed Metrics (Weighted) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    summary_data = {
        'model_name': [output_prefix],
        'top1_accuracy': [top1_acc],
        'top2_accuracy': [top2_acc],
        'top3_accuracy': [top3_acc],
        'top4_accuracy': [top4_acc],
        'top5_accuracy': [top5_acc],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
        'closed_set_auc': [roc_auc],
        'num_test_samples': [len(y_true)]
    }
    df_summary = pd.DataFrame(summary_data)
    summary_filename = f"result_csv/{output_prefix}_metrics.csv"
    df_summary.to_csv(summary_filename, index=False)
    print(f"Saved metrics summary to: {summary_filename}")

    results_list = []
    for i in range(len(y_true)):
        is_correct_closed = (y_true[i] == pred_labels[i, 0].item())

        is_correct_eer = (
            pred_eer[i].item() == y_true[i]
        )

        is_correct_far = (
            pred_far[i].item() == y_true[i]
        )
        
        # Determine what to save for true_class_score
        # If prediction is wrong, this value shows how confident the model was about the RIGHT answer
        t_score = y_true_scores[i]
        
        # Handle edge case: if t_score is -inf (meaning true class wasn't in gallery), set to -1
        if t_score == -float('inf'):
            t_score = -1.0

        row = {
            'filename': test_paths[i],
            'true_id': y_true[i],

            # Closed-set
            'pred_top1': pred_labels[i, 0].item(),
            'is_correct_closed': is_correct_closed,
            'score_top1': topk_scores[i, 0].item(),

            # Threshold-based (OPEN-SET)
            'pred_eer': pred_eer[i].item(),
            'pred_far': pred_far[i].item(),
            'accept_eer': bool(accept_eer[i].item()),
            'accept_far': bool(accept_far[i].item()),

            # True class score (analysis)
            'score_true_class': t_score,
        }
        results_list.append(row)
    open_set_acc_eer = np.mean([
        r['is_correct_closed'] and r['accept_eer']
        for r in results_list
    ]) * 100

    print(f"Open-set Accuracy (EER): {open_set_acc_eer:.2f}%")
    df_results = pd.DataFrame(results_list)
    
    # Optional: Calculate the margin (Confidence Gap)
    # This helps you sort by "Most Confident Errors"
    df_results['confidence_gap'] = df_results['score_top1'] - df_results['score_true_class']

    preds_filename = f"result_csv/{output_prefix}_predictions.csv"
    df_results.to_csv(preds_filename, index=False)
    return df_results, eer_thr

def compute_identification_threshold(
    sim_matrix,
    test_labels_true,
    gal_labels,
    output_prefix,
    target_far=0.01
):
    """
    Identification threshold using FAR–FRR intersection (EER).
    This definition follows biometric verification standards.
    """

    # --- Positive & Negative scores ---
    label_mask = test_labels_true.unsqueeze(1) == gal_labels.unsqueeze(0)

    pos_sim = sim_matrix.clone()
    pos_sim[~label_mask] = -float('inf')
    pos_scores = pos_sim.max(dim=1)[0].cpu().numpy()

    neg_sim = sim_matrix.clone()
    neg_sim[label_mask] = -float('inf')
    neg_scores = neg_sim.max(dim=1)[0].cpu().numpy()

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([
        np.ones_like(pos_scores),
        np.zeros_like(neg_scores)
    ])

    # --- ROC ---
    fpr, tpr, thresholds = roc_curve(labels, scores)

    far = fpr
    frr = 1 - tpr

    # Sort by threshold
    order = np.argsort(thresholds)
    thr = thresholds[order]
    far = far[order]
    frr = frr[order]

    diff = far - frr
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(idx) == 0:
        # fallback (rare, but safe)
        eer_idx = np.argmin(np.abs(diff))
        eer_thr = thr[eer_idx]
        eer = (far[eer_idx] + frr[eer_idx]) / 2
    else:
        i = idx[0]
        x0, x1 = thr[i], thr[i + 1]
        y0, y1 = diff[i], diff[i + 1]

        eer_thr = x0 - y0 * (x1 - x0) / (y1 - y0)
        eer = np.interp(eer_thr, thr, far)

    # FAR-based threshold (optional secondary operating point)
    far_idx = np.where(fpr <= target_far)[0][-1]
    far_threshold = thresholds[far_idx]

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(thr, far, label="FAR", linewidth=2)
    plt.plot(thr, frr, label="FRR", linewidth=2)

    plt.scatter(eer_thr, eer, color="red", zorder=5, label=f"EER = {eer:.4f}")
    plt.axvline(eer_thr, color="red", linestyle="--", alpha=0.7)

    plt.yscale("log")
    plt.ylim(1e-2, 1)
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR–FRR Intersection (EER)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"result_csv/{output_prefix}_eer_intersection.png")
    plt.close()

    print("=" * 30)
    print("Identification Thresholds")
    print(f"EER (intersection): {eer:.4f}")
    print(f"EER Threshold     : {eer_thr:.4f}")
    print(f"FAR@{target_far} Threshold: {far_threshold:.4f}")
    print("=" * 30)

    return eer_thr, far_threshold


def evaluate_open_set(
    test_embeddings,
    gallery_embeddings,
    threshold,          # <-- EER threshold
    output_prefix,
    device="cuda",
    top_k=5
):
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # -----------------------------
    # Prepare gallery
    # -----------------------------
    gal_feats, gal_labels = [], []

    for label, emb_list in gallery_embeddings.items():
        for emb in emb_list:
            gal_feats.append(emb)
            gal_labels.append(label)

    gal_feats = F.normalize(torch.stack(gal_feats), dim=1).to(device)
    gal_labels = torch.tensor(gal_labels).to(device)

    # -----------------------------
    # Prepare test
    # -----------------------------
    test_feats, test_labels, test_paths = [], [], []

    for label, emb_list in test_embeddings.items():
        for emb, path in emb_list:
            test_feats.append(emb)
            test_labels.append(label)  # unknown = -1
            test_paths.append(path)

    test_feats = F.normalize(torch.stack(test_feats), dim=1).to(device)
    test_labels = torch.tensor(test_labels).to(device)

    # -----------------------------
    # Similarity
    # -----------------------------
    sim_matrix = torch.matmul(test_feats, gal_feats.T)
    topk_scores, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)

    top1_scores = topk_scores[:, 0]
    pred_labels = gal_labels[topk_indices[:, 0]]

    # Accept / Reject using EER threshold
    accept = top1_scores >= threshold
    pred_open = pred_labels.clone()
    pred_open[~accept] = -1

    # -----------------------------
    # Per-sample results
    # -----------------------------
    rows = []
    for i in range(len(test_labels)):
        true_id = test_labels[i].item()
        pred_id = pred_open[i].item()
        score = top1_scores[i].item()

        is_unknown = (true_id == -1)

        if is_unknown:
            is_correct = not accept[i]
        else:
            is_correct = accept[i] and (pred_id == true_id)

        rows.append({
            "filename": test_paths[i],
            "true_id": true_id,
            "pred_id": pred_id,
            "score": score,
            "accepted": bool(accept[i]),
            "is_unknown": is_unknown,
            "correct_open": is_correct
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"result_csv/{output_prefix}_open_set_samples.csv", index=False)


    # Metrics (Known / Unknown)
    # -----------------------------
    known = df[df.true_id != -1]
    unknown = df[df.true_id == -1]

    # ---- Known ID metrics ----
    known_total = len(known)

    known_correct = (
        (known.accepted == True) &
        (known.pred_id == known.true_id)
    ).sum()

    known_false_reject = (
        known.accepted == False
    ).sum()

    known_misid = (
        (known.accepted == True) &
        (known.pred_id != known.true_id)
    ).sum()

    known_acc = known_correct / known_total * 100 if known_total > 0 else np.nan
    known_frr = known_false_reject / known_total * 100 if known_total > 0 else np.nan
    known_mir = known_misid / known_total * 100 if known_total > 0 else np.nan

    # ---- Unknown metrics ----
    unknown_total = len(unknown)

    far = unknown.accepted.mean() * 100 if unknown_total > 0 else np.nan
    trr = 100 - far if not np.isnan(far) else np.nan


    # -----------------------------
    # ROC (threshold-free)
    # -----------------------------
    y_true = (df.true_id != -1).astype(int)  # 1 = known
    y_score = df.score.values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("True Accept Rate (TAR)")
    plt.title("Open-Set ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"result_csv/{output_prefix}_open_set_roc.png")
    plt.close()

    # -----------------------------
    # Summary CSV (paper-ready)
    # -----------------------------
    summary = pd.DataFrame([{
        "threshold_type": "EER (FAR=FRR)",
        "threshold": threshold,

        # Known ID
        "known_id_accuracy_%": known_acc,
        "known_false_reject_rate_%": known_frr,
        "known_misidentification_rate_%": known_mir,

        # Unknown ID
        "false_accept_rate_%": far,
        "true_reject_rate_%": trr,

        # Threshold-free
        "open_set_roc_auc": roc_auc
    }])


    summary.to_csv(
        f"result_csv/{output_prefix}_open_set_summary.csv",
        index=False
    )

    # Console
    print("=" * 40)
    print("OPEN-SET IDENTIFICATION (EER THRESHOLD)")
    print(f"Threshold                 : {threshold:.4f}")
    print(f"Known-ID Accuracy         : {known_acc:.2f}%")
    print(f"Known False Reject Rate   : {known_frr:.2f}%")
    print(f"Known Mis-ID Rate         : {known_mir:.2f}%")
    print(f"False Accept Rate (UNK)   : {far:.2f}%")
    print(f"True Reject Rate  (UNK)   : {trr:.2f}%")
    print(f"ROC AUC                   : {roc_auc:.4f}")
    print("=" * 40)

    return summary, df


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.backbone == 'resnet':
    model = Network_Resnet(args.attention).to(device)
else:
    model = Network_ConvNext(args.backbone, args.attention).to(device)
model.load_state_dict(torch.load(f"model/model/{args.backbone}/{args.model}.pt", map_location=device))
model.eval()

result = []

with torch.no_grad():
    #gallery
    gallery_embeddings = {}

    for i, (image, label, _) in enumerate(gallery_loader):
        img_set1 = image.to(device)
        output = model(img_set1)

        for idx, item in enumerate(label):
            lbl = item.item()
            emb = output[idx]
            if lbl not in gallery_embeddings:
                gallery_embeddings[lbl] = [emb]
            else:
                gallery_embeddings[lbl].append(emb)

    test_embeddings = {}

    for (img, dog_id, img_path) in val_loader:
        img, dog_id = img.to(device), dog_id.to(device)
        emb = model(img)

        for idx2, item2 in enumerate(dog_id):
            lbl2 = item2.item()
            emb2 = emb[idx2]
            if lbl2 not in test_embeddings:
                test_embeddings[lbl2] = [(emb2, img_path[idx2])]
            else:
                test_embeddings[lbl2].append((emb2, img_path[idx2]))

    test_embeddings_unknown = {}

    for (img, dog_id, img_path) in val_loader_unknown:
        img, dog_id = img.to(device), dog_id.to(device)
        emb = model(img)

        for i in range(len(dog_id)):
            lbl = dog_id[i].item()
            emb_i = emb[i]
            if lbl not in test_embeddings_unknown:
                test_embeddings_unknown[lbl] = [(emb_i, img_path[i])]
            else:
                test_embeddings_unknown[lbl].append((emb_i, img_path[i]))

output_prefix = f"{args.output}_{args.resolution}"
df_emb_eval, eer_thr = evaluate_embedding_metrics(test_embeddings, gallery_embeddings, output_prefix, top_k=5)

df_open_set = evaluate_open_set(test_embeddings_unknown, gallery_embeddings, eer_thr, output_prefix, top_k=5)

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
import math

# --- Configuration ---
CLASSES_PER_PLOT = 45
random_seed = 42 
# ---------------------

print("Preparing data for Sorted Segmented t-SNE...")

# 1. Flatten the dictionary into arrays (Full Dataset)
embeddings_list = []
labels_list = []
paths_list = []   # OPTIONAL but very useful later

for label, emb_items in test_embeddings.items():
    for emb, path in emb_items:
        embeddings_list.append(emb.cpu().numpy())
        labels_list.append(label)
        paths_list.append(path)

X_full = np.array(embeddings_list)
y_full = np.array(labels_list)
paths_full = np.array(paths_list)

# 2. Get unique classes and SORT them numerically
unique_classes = np.unique(y_full)
unique_classes.sort() # <--- CHANGED: Forces strictly numerical order (1, 2, 3...)

# 3. Calculate number of batches needed
num_plots = math.ceil(len(unique_classes) / CLASSES_PER_PLOT)
print(f"Total unique classes: {len(unique_classes)}")
print(f"Splitting into {num_plots} plots with approx {CLASSES_PER_PLOT} sorted classes each.")

# --- Main Loop: Process one batch at a time ---
for i in range(num_plots):
    # Determine which classes belong to this batch
    start_idx = i * CLASSES_PER_PLOT
    end_idx = min((i + 1) * CLASSES_PER_PLOT, len(unique_classes))
    batch_classes = unique_classes[start_idx:end_idx]
    
    # Get the range for the filename (e.g., "IDs_0_to_44")
    range_str = f"IDs_{int(batch_classes[0])}_to_{int(batch_classes[-1])}"
    
    print(f"\n--- Processing Batch {i+1}/{num_plots} ({range_str}) ---")

    # 4. Filter the dataset to keep ONLY samples belonging to current batch classes
    mask = np.isin(y_full, batch_classes)
    X_batch = X_full[mask]
    y_batch = y_full[mask]
    
    n_samples_batch = len(X_batch)
    if n_samples_batch < 5:
        print("Batch too small, skipping.")
        continue

    # 5. Run t-SNE on this specific batch subset
    safe_perplexity = min(30, n_samples_batch - 5) 
    
    print(f"Running t-SNE on batch of {n_samples_batch} samples...")
    tsne = TSNE(
        n_components=2, 
        verbose=0, 
        perplexity=safe_perplexity, 
        metric='cosine', 
        init='random',
        learning_rate='auto',
        random_state=random_seed
    )
    tsne_results_batch = tsne.fit_transform(X_batch)

    # 6. Plotting the batch
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = cm.rainbow(np.linspace(0, 1, len(batch_classes)))

    for class_id, color in zip(batch_classes, colors):
        indices = y_batch == class_id
        x_coords = tsne_results_batch[indices, 0]
        y_coords = tsne_results_batch[indices, 1]
        
        # Plot dots
        ax.scatter(x_coords, y_coords, c=[color], alpha=0.6, s=35)
        
        # Plot Centroid Label
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        
        text = ax.text(
            centroid_x, centroid_y, str(int(class_id)), 
            fontsize=10, fontweight='bold', ha='center', va='center', color='black'
        )
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

    ax.set_title(f't-SNE Visualization - {range_str}', fontsize=16)
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')

    plt.tight_layout()

    folder_path = f"t-SNE/{args.output}"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save with ID range in filename for easier sorting in your folder
    save_filename = f"{folder_path}/{args.output}_tsne_{i+1:02d}_{range_str}.png"
    plt.savefig(save_filename, dpi=200)
    plt.close(fig) 
    print(f"Saved plot to {save_filename}")

print("\nAll t-SNE batches completed.")