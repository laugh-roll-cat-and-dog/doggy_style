import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import argparse
from sklearn.metrics import accuracy_score

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
parser.add_argument('-o', '--output', type=str, metavar="output",
                    help='output model filename')
parser.add_argument('-p', '--pretrained', choices=['n', 'coco', 'stand'], type=str, metavar="pretrained",
                    help='pre-trained model')
parser.add_argument('-e', '--epoch', type=int, metavar="epoch",
                    help='number of epoch')
parser.add_argument('-c', type=int, metavar="class",
                    help='number of class')
parser.add_argument('-l', '--loss', choices=['a', 's', 'as'], type=str, metavar="loss",
                    help='loss function')
parser.add_argument('-d', '--dataset', choices=['face', 'nose'], type=str, metavar="dataset",
                    help='what dataset to use')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## Data
train_transforms_1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms_2 = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    train_dataset_1 = DogDataset(csv_file='train_split_mixed_sorted.csv', transform=train_transforms_2)
    train_dataset_2 = DogDataset(csv_file='train_split_mixed_sorted.csv', transform=val_transforms)
    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    val_dataset = DogDataset(csv_file='test_split_mixed_sorted.csv', transform=val_transforms)

else:
    class DogDataset(Dataset):
        def __init__(self, csv_file, root_dir="", transform=None):
            self.df = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, index):
            # Read row
            img_name = self.df.loc[index, 'image']
            label = self.df.loc[index, 'dog_id']
            img_name = img_name.replace('\\', '/')
            img_path = os.path.join(self.root_dir, img_name) if self.root_dir else img_name

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Error loading image: {img_path} — {e}")

            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        
    train_dataset_1 = DogDataset(csv_file='dogNose_train.csv', transform=train_transforms_2)
    train_dataset_2 = DogDataset(csv_file='dogNose_train.csv', transform=val_transforms)
    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    val_dataset = DogDataset(csv_file='dogNose_test.csv', transform=val_transforms)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=6)

## NN
class Network(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        resnet = models.resnet50()
        if args.pretrained == 'coco':
            resnet.load_state_dict((torch.load("byol_coco.pt", map_location=device)))
        elif args.pretrained == 'stand':
            resnet.load_state_dict((torch.load("byol_stan.pt", map_location=device)))
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.extra_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.cam = ChannelAttentionModule()
        self.pam = PositionAttentionModule(512)
        self.dam = DualAttentionModule(in_channels=512)  
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.sam = Self_Attention(512)
        self.bam = BAM(512)
        in_chan = 0
        if args.attention == 'sb':
            in_chan = 1024
        elif args.attention == 'dsb':
            in_chan = 2048
        if args.bf == 't':
            in_chan += 512
        self.orchesta = FeatureFusionModule(in_chan, 3*512, args.attention, args.bf)

        self.fc = nn.Linear(3*512, embedding_dim, bias=False)  
        self.bn = nn.BatchNorm1d(embedding_dim)

    def embed(self, x):
        x = self.backbone(x)
        x = self.extra_layers(x)   
        if args.attention == 'd':
            x = self.dam(x)                 
        else:
            x = self.orchesta(self.sam(x), self.bam(x), self.cam(x), self.pam(x), x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)      
        x = self.fc(x)             
        x = self.bn(x)

        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img):
        emb = self.embed(img)
        return emb

## Scheduler
def lr_lambda(epoch):
    if epoch < args.epoch/2:
        return 1.0
    else:
        return max(0.0, 1.0 - (epoch - (args.epoch/2)) / (args.epoch/2))

# Eval fn
def evaluate(model, val_loader, arcface_loss, soft_triple_loss_1, soft_triple_loss_2, soft_triple_loss_3, ce_loss, device):
    model.eval()
    if arcface_loss is not None:
        arcface_loss.eval()
    if soft_triple_loss_1 is not None:
        soft_triple_loss_1.eval()
        soft_triple_loss_2.eval()
        soft_triple_loss_3.eval()

    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img, dog_id in val_loader:
            img, dog_id = img.to(device), dog_id.to(device)
            output = model(img)

            loss = 0.0
            combined_logits = 0.0

            if args.loss == 'a':
                logits_a = arcface_loss(output, dog_id)
                #print("arcface logit:", sum(logits_a))
                loss_a = ce_loss(logits_a, dog_id)
                # print("arcface loss:", loss_a)
                loss += loss_a
                combined_logits += torch.softmax(logits_a, dim=1)

            if soft_triple_loss_1 is not None:
                
                logits_s_1 = soft_triple_loss_1(output)
                loss_s_1 = soft_triple_loss_1.loss(logits_s_1, dog_id)
                logits_s_2 = soft_triple_loss_2(output)
                loss_s_2 = soft_triple_loss_2.loss(logits_s_2, dog_id)
                logits_s_3 = soft_triple_loss_3(output)
                loss_s_3 = soft_triple_loss_3.loss(logits_s_3, dog_id)
                loss += loss_s_1 + loss_s_2 + loss_s_3
                logits_s = logits_s_1 + logits_s_2 + logits_s_3
                combined_logits += torch.softmax(logits_s, dim=1)

            # if arcface_loss is not None and soft_triple_loss_1 is not None:
            #     combined_logits /= 2.0

            val_loss += loss.item()

            preds = torch.argmax(combined_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(dog_id.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc

## Train
def train():
    model = Network().to(device)
    
    if args.pretrained == 'coco' or args.pretrained == 'stand':
        for param in model.backbone.parameters():
            param.requires_grad = False

    arcface_loss, soft_triple_loss_1, soft_triple_loss_2, soft_triple_loss_3 = None, None, None, None
    if 'a' in args.loss:
        arcface_loss = ArcFace(1024, int(args.c)).to(device)
    if 's' in args.loss:
        soft_triple_loss_1 = SoftTriple(8, 0.1, 0.1, 0.03, 1024, int(args.c), 2).to(device)
        soft_triple_loss_2 = SoftTriple(10, 0.1, 0.1, 0.02, 1024, int(args.c), 2).to(device)
        soft_triple_loss_3 = SoftTriple(12, 0.1, 0.1, 0.01, 1024, int(args.c), 2).to(device)

    ce_loss = nn.CrossEntropyLoss()

    param_groups = [{"params": model.parameters()}]
    if arcface_loss is not None:
        param_groups.append({"params": arcface_loss.parameters()})
    if soft_triple_loss_1 is not None:
        param_groups.append({"params": soft_triple_loss_1.parameters()})
        param_groups.append({"params": soft_triple_loss_2.parameters()})
        param_groups.append({"params": soft_triple_loss_3.parameters()})
    optimizer = Adam(param_groups, lr=0.00035, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    num_epochs = args.epoch

    for epoch in range(num_epochs):
        model.train()
        if 'a' in args.loss:
            arcface_loss.train()
        if 's' in args.loss:
            soft_triple_loss_1.train()
            soft_triple_loss_2.train()
            soft_triple_loss_3.train()
        training_loss = 0
        correct, total = 0, 0

        for img, dog_id in train_loader:
            img_set = Variable(img.to(device))
            id_set = Variable(dog_id.to(device))

            output = model(img_set)

            loss = 0
            combined_logits = 0.0
            if 'a' in args.loss:
                logits_a = arcface_loss(output, id_set)
                #print(logits_a)
                loss_a = ce_loss(logits_a, id_set)
                loss += loss_a
                if 's' not in args.loss:
                    combined_logits += torch.softmax(logits_a, dim=1)
            if 's' in args.loss:
                logits_s_1 = soft_triple_loss_1(output)
                loss_s_1 = soft_triple_loss_1.loss(logits_s_1, id_set)
                logits_s_2 = soft_triple_loss_2(output)
                loss_s_2 = soft_triple_loss_2.loss(logits_s_2, id_set)
                logits_s_3 = soft_triple_loss_3(output)
                loss_s_3 = soft_triple_loss_3.loss(logits_s_3, id_set)
                loss += loss_s_1 + loss_s_2 + loss_s_3
                # loss = awl(loss_s_1, loss_s_2, loss_s_3)
                logits_s = logits_s_1 + logits_s_2 + logits_s_3
                combined_logits += torch.softmax(logits_s, dim=1)
            # if arcface_loss is not None and soft_triple_loss_1 is not None:
            #     loss /= 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            preds = torch.argmax(combined_logits, dim=1).to(torch.device("cpu"))
            # print(preds[:10], id_set[:10])
            correct += (preds == dog_id).sum().item()
            total += dog_id.size(0)

        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {training_loss/len(train_loader):.4f}, Train Acc: {train_acc * 100:.4f}")
        val_loss, val_acc = evaluate(model, val_loader, arcface_loss, soft_triple_loss_1, soft_triple_loss_2, soft_triple_loss_3, ce_loss, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), f"{args.output}.pt")
    if arcface_loss is not None:
        torch.save(arcface_loss.state_dict(), f"{args.output}_arcface.pt")
    if soft_triple_loss_1 is not None:
        torch.save(soft_triple_loss_1.state_dict(), f"{args.output}_soft_ensemble_1.pt")
        torch.save(soft_triple_loss_2.state_dict(), f"{args.output}_soft_ensemble_2.pt")
        torch.save(soft_triple_loss_3.state_dict(), f"{args.output}_soft_ensemble_3.pt")

train()