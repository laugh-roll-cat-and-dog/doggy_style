import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def verification(model, pairs_csv_path, transform, device):
    """
    Docstring for verification
    
    :param model: Neural network model for verification
    :param pairs_csv_path: Path to CSV file containing image pairs
    :param transform: Transformations to apply to images
    :param device: Device (CPU/GPU) to run the model on
    """
    if not os.path.exists(pairs_csv_path):
        print(f"ไม่เจอไฟล์ที่ระบุ: {pairs_csv_path}")
        print("กรุณาตรวจสอบเส้นทางและลองใหม่อีกครั้ง")
        return
    
    pairs_df = pd.read_csv(pairs_csv_path)
    scores = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i, row in pairs_df.iterrows():
            img1_path = row['image1_path']
            img2_path = row['image2_path']
            label = int(row['label'])

            try:
                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')

                img1 = transform(img1).unsqueeze(0).to(device)
                img2 = transform(img2).unsqueeze(0).to(device)

                emb1 = model(img1)
                emb2 = model(img2)

                sim = F.cosine_similarity(emb1, emb2).item()
                scores.append(sim)
                labels.append(label)
            except Exception as e:
                print(f"ไม่สามารถประมวลผลคู่ภาพ: {img1_path}, {img2_path}. ข้ามคู่ภาพนี้. ข้อผิดพลาด: {e}")
                continue
    
    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    #EER (Equal Error Rate)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    def get_tar_at_far(target_far):
        idx = np.where(fpr <= target_far)[0][-1]
        return tpr[idx]

    tar_01 = get_tar_at_far(0.01)
    tar_001 = get_tar_at_far(0.001)

    print("="*30)
    print("ผลการตรวจสอบ:")
    print(f"AUC Score:  {roc_auc:.4f}")
    print(f"EER:        {eer:.4f}")
    print(f"Target Rate at FAR=0.01: {tar_01:.4f}")
    print(f"Target Rate at FAR=0.001: {tar_001:.4f}")

    plt.figure(figsize=(10, 6))

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    plt.hist(pos_scores, bins=50, alpha=0.5, label='Positive Pairs', color='g', density=True)
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Negative Pairs', color='r', density=True)

    plt.axvline(x=thresholds[np.argmax(tpr - fpr)], color='b', linestyle='--', label='Optimal Threshold')

    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('similarity_score_distribution.png')
    print("บันทึกกราฟ Histogram ไว้ที่: verification_score_dist.png")

    return roc_auc, eer
