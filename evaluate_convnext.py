import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import math
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os

from network.network import Network_Resnet, Network_ConvNext

from loss.arcface import ArcFace
from loss.softTriple import SoftTriple

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--attention', default='sb', choices=['s', 'b', 'sb', 'n'], type=str, metavar="attention",
                    help='attention module')
parser.add_argument('-b', '--backbone', choices=['dino', 'v2', 'resnet'], type=str, metavar="backbone",
                    help='backbone')
parser.add_argument('-c', type=int, metavar="class",
                    help='number of class')
parser.add_argument('-m', '--model', type=str, metavar="model",
                    help=' model filename')
# parser.add_argument('-ah', '--arcface', type=str, metavar="arcface",
#                     help='arcface classifier head filename')
# parser.add_argument('-sh', '--softtriple', type=str, metavar="softtriple",
#                     help='softtriple classifier head filename')
parser.add_argument('-o', '--output', type=str, metavar="output",
                    help='output csv filename')
parser.add_argument('-l', '--loss', choices=['a', 's', 'as'], type=str, metavar="loss",
                    help='loss function')
parser.add_argument('-d', '--dataset', choices=['face', 'nose', 'nose_old'], type=str, metavar="dataset",
                    help='what dataset to use')
args = parser.parse_args()

## Data

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
                    if label < 38:
                        continue
                except:
                    continue

                for root, dirs, files in os.walk(class_dir):
                    if os.path.basename(root) != split:
                        continue

                    for i, fname in enumerate(files):
                        fpath = os.path.join(root, fname)
                        if os.path.isfile(fpath):
                            if split == 'train' and i >= 5:
                                continue
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

            return image, label
    val_dataset = DogDataset('crop', 'test', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    gallery_dataset = DogDataset('crop', transform=val_transforms)
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
    for label, emb_list in test_embeddings.items():
        for emb in emb_list:
            test_feats.append(emb)
            test_labels.append(label)

    test_feats = torch.stack(test_feats).to(device)
    test_feats = F.normalize(test_feats, p=2, dim=1)
    test_labels_true = torch.tensor(test_labels).to(device)

    # 1. Compute Similarity Matrix
    sim_matrix = torch.matmul(test_feats, gal_feats.T)

    # 2. Get Top-K Predictions
    topk_scores, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)
    pred_labels = gal_labels[topk_indices]

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
        'num_test_samples': [len(y_true)]
    }
    df_summary = pd.DataFrame(summary_data)
    summary_filename = f"result_csv/{output_prefix}_metrics.csv"
    df_summary.to_csv(summary_filename, index=False)
    print(f"Saved metrics summary to: {summary_filename}")

    results_list = []
    for i in range(len(y_true)):
        is_correct = (y_true[i] == pred_labels[i, 0].item())
        
        # Determine what to save for true_class_score
        # If prediction is wrong, this value shows how confident the model was about the RIGHT answer
        t_score = y_true_scores[i]
        
        # Handle edge case: if t_score is -inf (meaning true class wasn't in gallery), set to -1
        if t_score == -float('inf'):
            t_score = -1.0

        row = {
            'true_id': y_true[i],
            'pred_top1': pred_labels[i, 0].item(),
            'is_correct': is_correct,
            'score_top1': topk_scores[i, 0].item(),      # Score of the predicted class
            'score_true_class': t_score,                 # Score of the actual correct class
            'pred_top2': pred_labels[i, 1].item() if top_k >=2 else -1,
            'pred_top3': pred_labels[i, 2].item() if top_k >=3 else -1,
            'pred_top4': pred_labels[i, 3].item() if top_k >=4 else -1,
            'pred_top5': pred_labels[i, 4].item() if top_k >=5 else -1,
        }
        results_list.append(row)
    
    df_results = pd.DataFrame(results_list)
    
    # Optional: Calculate the margin (Confidence Gap)
    # This helps you sort by "Most Confident Errors"
    df_results['confidence_gap'] = df_results['score_top1'] - df_results['score_true_class']

    preds_filename = f"result_csv/{output_prefix}_predictions.csv"
    df_results.to_csv(preds_filename, index=False)
    return df_results

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

    for i, (image, label) in enumerate(gallery_loader):
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

    for (img, dog_id) in val_loader:
        img, dog_id = img.to(device), dog_id.to(device)
        emb = model(img)

        for idx2, item2 in enumerate(dog_id):
            lbl2 = item2.item()
            emb2 = emb[idx2]
            if lbl2 not in test_embeddings:
                test_embeddings[lbl2] = [emb2]
            else:
                test_embeddings[lbl2].append(emb2)

df_emb_eval = evaluate_embedding_metrics(test_embeddings, gallery_embeddings, args.output, top_k=5)

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

for label, emb_tensors in test_embeddings.items():
    for emb in emb_tensors:
        embeddings_list.append(emb.cpu().numpy())
        labels_list.append(label)

X_full = np.array(embeddings_list)
y_full = np.array(labels_list)

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