import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os

from attention.BAM import BAM
from attention.DAM import ChannelAttentionModule, PositionAttentionModule, DualAttentionModule
from attention.SAM import Self_Attention
from attention.SEblock import FeatureFusionModule

from loss.arcface import ArcFace
from loss.softTriple import SoftTriple

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--attention', default='d', choices=['d', 'sb', 'dsb'], type=str, metavar="attention",
                    help='attention module')
parser.add_argument('-bf', choices=['t', 'f'], type=str, metavar="bf",
                    help='concat backbone feature maps to last feature maps')
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
        def __init__(self,csv_file, transform=None):
            self.df = pd.read_csv(csv_file)
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, index):
            img_path = self.df.loc[index,'filename']
            img_path = os.path.join("dog_nose_2022", img_path)
            label = self.df.loc[index,'new_dog_id']
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            
            return image, label
    val_dataset = DogDataset(csv_file='validation.csv', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    gallery_dataset = DogDataset(csv_file='training.csv', transform=val_transforms)
    gallery_loader = DataLoader(gallery_dataset, batch_size=16,shuffle=False)

## NN
class Network(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        model_name = "facebook/convnextv2-tiny-1k-224"
        base_model = ConvNextV2ForImageClassification.from_pretrained(model_name)

        self.backbone = base_model.convnextv2
        num_features = base_model.classifier.in_features
        for name, param in self.backbone.named_parameters():
            if 'stages.3' in name or 'convnextv2.layernorm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.cam = ChannelAttentionModule()
        self.pam = PositionAttentionModule(num_features)
        self.dam = DualAttentionModule(in_channels=num_features)
        self.sam = Self_Attention(num_features)
        self.bam = BAM(num_features)
        
        in_chan = 0
        if args.attention == 'sb' or args.attention == 'd':
            in_chan = num_features * 2
        elif args.attention == 'dsb':
            in_chan = num_features * 4
        else:
            in_chan = num_features

        if args.bf == 't':
            in_chan += num_features

        self.orchestra = FeatureFusionModule(
            in_chan=in_chan,
            out_chan=2 * num_features,
            attention=args.attention,
            bf=args.bf
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2 * num_features, embedding_dim, bias=False)
        self.ln = nn.LayerNorm(embedding_dim)

    def extract_backbone(self, x):
        out = self.backbone(x)
        features = out.last_hidden_state
        return features

    def embed(self, x):
        feat = self.extract_backbone(x)

        if args.attention == 'd':
            att1 = self.cam(feat)
            att2 = self.pam(feat)
            fused = self.orchestra(cam=att1, pam=att2, x=feat)
        elif args.attention == 'sb':
            att1 = self.sam(feat)
            att2 = self.bam(feat)
            fused = self.orchestra(fsp=att1, fcp=att2, x=feat)
        elif args.attention == 'dsb':
            att1 = self.cam(feat)
            att2 = self.pam(feat)
            att3 = self.sam(feat)
            att4 = self.bam(feat)
            fused = self.orchestra(cam=att1, pam=att2, fsp=att3, fcp=att4, x=feat)
        else:
            fused = feat

        fused = self.gap(fused)
        fused = fused.view(fused.size(0), -1)

        x = self.fc(fused)
        x = self.ln(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img):
        return self.embed(img)

def evaluate_embedding_metrics(test_embeddings, gallery_embeddings,output_prefix, top_k=5):
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

    sim_matrix = torch.matmul(test_feats, gal_feats.T)

    topk_scores, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)

    pred_labels = gal_labels[topk_indices]

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
        row = {
            'true_id': y_true[i],
            'pred_top1': pred_labels[i, 0].item(),
            'pred_top2': pred_labels[i, 1].item() if top_k >=2 else -1,
            'pred_top3': pred_labels[i, 2].item() if top_k >=3 else -1,
            'pred_top4': pred_labels[i, 3].item() if top_k >=4 else -1,
            'pred_top5': pred_labels[i, 4].item() if top_k >=5 else -1,
            'score_top1': topk_scores[i, 0].item()
        }
        results_list.append(row)
    
    df_results = pd.DataFrame(results_list)
    df_results['is_correct'] = df_results['true_id'] == df_results['pred_top1']
    preds_filename = f"result_csv/{output_prefix}_predictions.csv"
    df_results.to_csv(preds_filename, index=False)
    return df_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network().to(device)
model.load_state_dict(torch.load(f"{args.model}.pt", map_location=device))
model.eval()

arcface, softtriple_1 = None, None
if 'a' in args.loss:
    arcface = ArcFace(1024, int(args.c)).to(device)
    arcface.load_state_dict(torch.load(f"{args.model}_arcface.pt", map_location=device))
    arcface.eval()

if 's' in args.loss:
    
    softtriple_1 = SoftTriple(8, 0.1, 0.1, 0.03, 1024, int(args.c), 2).to(device)
    softtriple_2 = SoftTriple(10, 0.1, 0.1, 0.02, 1024, int(args.c), 2).to(device)
    softtriple_3 = SoftTriple(12, 0.1, 0.1, 0.01, 1024, int(args.c), 2).to(device)
    softtriple_1.load_state_dict(torch.load(f"{args.model}_soft_ensemble_1.pt", map_location=device))
    softtriple_2.load_state_dict(torch.load(f"{args.model}_soft_ensemble_2.pt", map_location=device))
    softtriple_3.load_state_dict(torch.load(f"{args.model}_soft_ensemble_3.pt", map_location=device))
    softtriple_1.eval()
    softtriple_2.eval()
    softtriple_3.eval()

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

        combined_logits = 0.0
        if arcface is not None:
            logits_a = arcface(emb, dog_id)
            combined_logits += torch.softmax(logits_a, dim=1)

        if softtriple_1 is not None:
            logits_s = softtriple_1(emb)
            logits_s += softtriple_2(emb)
            logits_s += softtriple_3(emb)
            combined_logits += torch.softmax(logits_s, dim=1)

        if arcface is not None and softtriple_1 is not None:
            combined_logits /= 2.0

        pred_prob = torch.softmax(combined_logits, dim=1)
        top5_prob, top5_idx = torch.topk(pred_prob, k=5, dim=1)

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