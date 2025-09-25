ResNet-50, BATCH SIZE = 16, LR = 0.00035, optimizer = Adam

# Data Augment
offline = 2x

# 1 

| Pretrained     | Attention | Last Feature Map     |
| -------------- | --------- | -------------------- |
| -              | DAM       | DAM + BB Feature Map |
| standford dogs | DAM       | DAM + BB Feature Map |
| Coco           | DAM       | DAM + BB Feature Map |

# 2

| Pretrained     | Attention | Last Feature Map |
| -------------- | --------- | ---------------- |
| -              | SAM+BAM   | SAM+BAM          |
| -              | SAM+BAM   | SAM+BAM+BB Feature Map         |
| standford dogs | SAM+BAM   | SAM+BAM          |
| standford dogs | SAM+BAM   | SAM+BAM+BB Feature Map          |
| Coco           | SAM+BAM   | SAM+BAM          |
| Coco           | SAM+BAM   | SAM+BAM+BB Feature Map          |

# 3

| Pretrained     | Attention   | Last Feature Map           |
| -------------- | ----------- | -------------------------- |
| -              | SAM+BAM+DAM | SAM+BAM+DAM                |
| -              | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |
| standford dogs | SAM+BAM+DAM | SAM+BAM+DAM                |
| standford dogs | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |
| Coco           | SAM+BAM+DAM | SAM+BAM+DAM                |
| Coco           | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |

# Evaluation method

1. Embedding Vector -> Threshold at EER point
2. Classification Head -> Avg. between Arc and Soft head
