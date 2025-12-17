import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def evaluate_embedding_metrics(test_embeddings, gallery_embeddings, top_k=5):
    print("--- Starting Embedding-Based Evaluation ---")
    
    # 1. Flatten Gallery Dictionary to Tensors
    # We create a big matrix of all gallery vectors
    gal_feats = []
    gal_labels = []
    for label, emb_list in gallery_embeddings.items():
        for emb in emb_list:
            gal_feats.append(emb)
            gal_labels.append(label)
    
    # Stack and Normalize Gallery (N_gallery, 1024)
    gal_feats = torch.stack(gal_feats).to(device)
    gal_feats = F.normalize(gal_feats, p=2, dim=1)
    gal_labels = torch.tensor(gal_labels).to(device)

    # 2. Flatten Query (Test) Dictionary to Tensors
    test_feats = []
    test_labels = []
    for label, emb_list in test_embeddings.items():
        for emb in emb_list:
            test_feats.append(emb)
            test_labels.append(label)
            
    # Stack and Normalize Query (N_test, 1024)
    test_feats = torch.stack(test_feats).to(device)
    test_feats = F.normalize(test_feats, p=2, dim=1)
    test_labels_true = torch.tensor(test_labels).to(device)

    # 3. Compute Cosine Similarity Matrix
    # Shape: (N_test, N_gallery)
    # This is much faster than loops
    sim_matrix = torch.matmul(test_feats, gal_feats.T)

    # 4. Get Top-K Scores and Indices
    # values: cosine scores, indices: index in gal_feats
    topk_scores, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)

    # 5. Convert Indices to actual Class Labels
    # We gather the labels from gal_labels using the indices found
    # Shape: (N_test, K)
    pred_labels = gal_labels[topk_indices]

    # --- METRIC CALCULATION ---
    
    # A. Top-K Accuracy
    # Check if true label is in the top K predictions
    # Expand true labels to (N_test, 1) to compare with (N_test, K)
    correct = pred_labels.eq(test_labels_true.view(-1, 1).expand_as(pred_labels))

    top1_acc = correct[:, 0].float().mean().item() * 100
    top5_acc = correct[:, :5].any(dim=1).float().mean().item() * 100

    print(f"Embedding Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Embedding Top-5 Accuracy: {top5_acc:.2f}%")

    # B. Detailed Classification Metrics (Precision, Recall, F1)
    # We use the Top-1 Prediction for standard classification metrics
    y_true = test_labels_true.cpu().numpy()
    y_pred = pred_labels[:, 0].cpu().numpy() # Take the 1st column (Top-1)

    # Calculate weighted metrics (accounts for class imbalance)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    print("\n--- Detailed Metrics (Weighted) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # C. Per-Class Report (Optional, good for finding weak classes)
    # unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    # if len(unique_classes) < 50: # Only print if not too huge
    #     print("\n--- Classification Report ---")
    #     print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    # 6. Save Results to CSV
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
    return df_results

# --- EXECUTE ---
# Call this function passing your existing dictionaries
df_emb_eval = evaluate_embedding_metrics(test_embeddings, gallery_embeddings, top_k=5)

# Save to CSV
df_emb_eval.to_csv(f"{args.output}_embedding_eval.csv", index=False)