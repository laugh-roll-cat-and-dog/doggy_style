ResNet-50, BATCH SIZE = 16, LR = 0.00035, optimizer = Adam

# Data Augment
offline = 2x

Loss |
-----|
Soft |
Arc |
Soft + Arc |


# 1 

|Output| Pretrained     | Attention | Last Feature Map     |Loss |
|------| -------------- | --------- | -------------------- |-----|
| 1.s  | -              | DAM       | DAM + BB Feature Map |Soft |
| 1.a  | -              | DAM       | DAM + BB Feature Map |Arc |
| 1.as | -              | DAM       | DAM + BB Feature Map |Soft + Arc |
| 2.s  | standford dogs | DAM       | DAM + BB Feature Map |Soft |
| 2.a  | standford dogs | DAM       | DAM + BB Feature Map |Arc |
| 2.as | standford dogs | DAM       | DAM + BB Feature Map |Soft + Arc |
| 3.s  | Coco           | DAM       | DAM + BB Feature Map |Soft |
| 3.a  | Coco           | DAM       | DAM + BB Feature Map |Arc |
| 3.as | Coco           | DAM       | DAM + BB Feature Map |Soft + Arc |

# 2

|Output| Pretrained     | Attention | Last Feature Map |Loss |
|------| -------------- | --------- | ---------------- |-----|
| 4.s  | -              | SAM+BAM   | SAM+BAM          |Soft |
| 4.a  | -              | SAM+BAM   | SAM+BAM          |Arc |
| 4.as | -              | SAM+BAM   | SAM+BAM          |Soft + Arc |
| 5.s  | -              | SAM+BAM   | SAM+BAM+BB Feature Map         |Soft |
| 5.a  | -              | SAM+BAM   | SAM+BAM+BB Feature Map         |Arc |
| 5.as | -              | SAM+BAM   | SAM+BAM+BB Feature Map         |Soft + Arc |
| 6.s  | standford dogs | SAM+BAM   | SAM+BAM          |Soft |
| 6.a  | standford dogs | SAM+BAM   | SAM+BAM          |Arc |
| 6.as | standford dogs | SAM+BAM   | SAM+BAM          |Soft + Arc |
| 7.s  | standford dogs | SAM+BAM   | SAM+BAM+BB Feature Map          |Soft |
| 7.a  | standford dogs | SAM+BAM   | SAM+BAM+BB Feature Map          |Arc |
| 7.as | standford dogs | SAM+BAM   | SAM+BAM+BB Feature Map          |Soft + Arc |
| 8.s  | Coco           | SAM+BAM   | SAM+BAM          |Soft |
| 8.a  | Coco           | SAM+BAM   | SAM+BAM          |Arc |
| 8.as | Coco           | SAM+BAM   | SAM+BAM          |Soft + Arc |
| 9.s  | Coco           | SAM+BAM   | SAM+BAM+BB Feature Map          |Soft |
| 9.a  | Coco           | SAM+BAM   | SAM+BAM+BB Feature Map          |Arc |
| 9.as | Coco           | SAM+BAM   | SAM+BAM+BB Feature Map          |Soft + Arc |

# 3

|Output| Pretrained     | Attention   | Last Feature Map           |Loss |
|------| -------------- | ----------- | -------------------------- |-----|
| 10.s  | -              | SAM+BAM+DAM | SAM+BAM+DAM                |Soft |
| 10.a  | -              | SAM+BAM+DAM | SAM+BAM+DAM                |Arc |
| 10.as | -              | SAM+BAM+DAM | SAM+BAM+DAM                |Soft + Arc |
| 11.s  | -              | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft |
| 11.a  | -              | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Arc |
| 11.as | -              | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft + Arc |
| 12.s  | standford dogs | SAM+BAM+DAM | SAM+BAM+DAM                |Soft |
| 12.a  | standford dogs | SAM+BAM+DAM | SAM+BAM+DAM                |Arc |
| 12.as | standford dogs | SAM+BAM+DAM | SAM+BAM+DAM                |Soft + Arc |
| 13.s  | standford dogs | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft |
| 13.a  | standford dogs | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Arc |
| 13.as | standford dogs | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft + Arc |
| 14.s  | Coco           | SAM+BAM+DAM | SAM+BAM+DAM                |Soft |
| 14.a  | Coco           | SAM+BAM+DAM | SAM+BAM+DAM                |Arc |
| 14.as | Coco           | SAM+BAM+DAM | SAM+BAM+DAM                |Soft + Arc |
| 15.s  | Coco           | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft |
| 15.a  | Coco           | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Arc |
| 15.as | Coco           | SAM+BAM+DAM | SAM+BAM+DAM+BB Feature Map |Soft + Arc |

# Evaluation method

1. Embedding Vector -> Threshold at EER point
2. Classification Head -> Avg. between Arc and Soft head
