ResNet-50, BATCH SIZE = 16, LR = 0.00035, optimizer = Adam

# Data Augment
offline = 2x

Loss |
-----|
Soft |
Arc |
Soft + Arc |


# 1 

| Pretrained     | Attention | Last Feature Map     |Loss |
| -------------- | --------- | -------------------- |-----|
| -              | DAM       | DAM + BB Feature Map |Soft |
| -              | DAM       | DAM + BB Feature Map |Arc |
| -              | DAM       | DAM + BB Feature Map |Soft + Arc |
| standford dogs | DAM       | DAM + BB Feature Map |Soft |
| standford dogs | DAM       | DAM + BB Feature Map |Arc |
| standford dogs | DAM       | DAM + BB Feature Map |Soft + Arc |
| Coco           | DAM       | DAM + BB Feature Map |Soft |
| Coco           | DAM       | DAM + BB Feature Map |Arc |
| Coco           | DAM       | DAM + BB Feature Map |Soft + Arc |

# 2

| Pretrained     | Attention | Last Feature Map |Loss |
| -------------- | --------- | ---------------- |-----|
| -              | SAM+BAM   | SAM+BAM          |Soft |
| -              | SAM+BAM   | SAM+BAM          |Arc |
| -              | SAM+BAM   | SAM+BAM          |Soft + Arc |
| -              | SAM+BAM   | SAM+BAM+BB Feature Map         |Soft |
| -              | SAM+BAM   | SAM+BAM+BB Feature Map         |Arc |
| -              | SAM+BAM   | SAM+BAM+BB Feature Map         |Soft + Arc |
| standford dogs | SAM+BAM   | SAM+BAM          |Soft |
| standford dogs | SAM+BAM   | SAM+BAM          |Arc |
| standford dogs | SAM+BAM   | SAM+BAM          |Soft + Arc |
| standford dogs | SAM+BAM   | SAM+BAM+BB Feature Map          |Soft |
| standford dogs | SAM+BAM   | SAM+BAM+BB Feature Map          |Arc |
| standford dogs | SAM+BAM   | SAM+BAM+BB Feature Map          |Soft + Arc |
| Coco           | SAM+BAM   | SAM+BAM          |Soft |
| Coco           | SAM+BAM   | SAM+BAM          |Arc |
| Coco           | SAM+BAM   | SAM+BAM          |Soft + Arc |
| Coco           | SAM+BAM   | SAM+BAM+BB Feature Map          |Soft |
| Coco           | SAM+BAM   | SAM+BAM+BB Feature Map          |Arc |
| Coco           | SAM+BAM   | SAM+BAM+BB Feature Map          |Soft + Arc |

# 3

| Pretrained     | Attention   | Last Feature Map           |Loss |
| -------------- | ----------- | -------------------------- |-----|
| -              | SAM+BAM+DAM | SAM+BAM+DAM                |Soft |
| -              | SAM+BAM+DAM | SAM+BAM+DAM                |Arc |
| -              | SAM+BAM+DAM | SAM+BAM+DAM                |Soft + Arc |
| -              | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft |
| -              | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Arc |
| -              | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft + Arc |
| standford dogs | SAM+BAM+DAM | SAM+BAM+DAM                |Soft |
| standford dogs | SAM+BAM+DAM | SAM+BAM+DAM                |Arc |
| standford dogs | SAM+BAM+DAM | SAM+BAM+DAM                |Soft + Arc |
| standford dogs | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft |
| standford dogs | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Arc |
| standford dogs | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft + Arc |
| Coco           | SAM+BAM+DAM | SAM+BAM+DAM                |Soft |
| Coco           | SAM+BAM+DAM | SAM+BAM+DAM                |Arc |
| Coco           | SAM+BAM+DAM | SAM+BAM+DAM                |Soft + Arc |
| Coco           | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft |
| Coco           | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Arc |
| Coco           | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft + Arc |

# Evaluation method

1. Embedding Vector -> Threshold at EER point
2. Classification Head -> Avg. between Arc and Soft head
