import os
import time
import json
import platform
import threading
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.autograd import Variable
import argparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from natsort import natsorted

from network.network import Network_ConvNext, Network_Resnet

from loss.arcface import ArcFace
from loss.softTriple import SoftTriple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


class GPUPowerMonitor:
    """Samples GPU power draw via NVML in a background thread and integrates
    it over time to estimate energy consumption (Joules).

    Falls back to a no-op when NVML or CUDA is unavailable, so the training
    script still runs on CPU-only or non-NVIDIA edge devices.
    """

    def __init__(self, device_index=0, sample_interval=0.5):
        self.device_index = device_index
        self.sample_interval = sample_interval
        self.samples = []
        self.running = False
        self.thread = None
        self.handle = None
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self.handle = None

    def _sample_loop(self):
        while self.running:
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append((time.time(), p_mw / 1000.0))
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def start(self):
        if self.handle is None:
            return
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def energy_joules(self):
        if len(self.samples) < 2:
            return 0.0
        e = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i][0] - self.samples[i - 1][0]
            avg_p = (self.samples[i][1] + self.samples[i - 1][1]) / 2.0
            e += avg_p * dt
        return e

    def avg_power_w(self):
        if not self.samples:
            return 0.0
        return sum(s[1] for s in self.samples) / len(self.samples)

    def peak_power_w(self):
        if not self.samples:
            return 0.0
        return max(s[1] for s in self.samples)


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_size_mb(module):
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def cpu_rss_mb():
    if not PSUTIL_AVAILABLE:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def gpu_static_info():
    info = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["name"] = torch.cuda.get_device_name(0)
        info["total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    if PYNVML_AVAILABLE and torch.cuda.is_available():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            cap = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            info["power_limit_w"] = cap
        except Exception:
            pass
    return info


def profile_inference(model, loader, device, max_batches=50, warmup=5):
    """Measure latency, peak memory and (if NVML is up) energy per image.
    Stands in for the edge-deployment profile the reviewer asked for: numbers
    are reported per-image so they translate to mobile / IoT budgets.
    """
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    monitor = GPUPowerMonitor(sample_interval=0.05)

    n_images = 0
    batches_done = 0
    with torch.no_grad():
        # Warmup so we don't bill JIT / cuDNN tuning to the measurement.
        for i, (img, _) in enumerate(loader):
            if i >= warmup:
                break
            img = img.to(device)
            _ = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()

        monitor.start()
        t0 = time.perf_counter()
        for img, _ in loader:
            img = img.to(device)
            _ = model(img)
            n_images += img.size(0)
            batches_done += 1
            if batches_done >= max_batches:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        monitor.stop()

    elapsed = t1 - t0
    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if device.type == "cuda" else None)
    energy_j = monitor.energy_joules()

    return {
        "images_processed": n_images,
        "elapsed_s": elapsed,
        "latency_ms_per_image": (elapsed / n_images) * 1000.0 if n_images else None,
        "throughput_imgs_per_s": n_images / elapsed if elapsed > 0 else None,
        "peak_gpu_mem_mb": peak_mem_mb,
        "gpu_energy_j": energy_j if energy_j > 0 else None,
        "energy_mj_per_image": (energy_j / n_images) * 1000.0
                                if (energy_j > 0 and n_images) else None,
        "avg_gpu_power_w": monitor.avg_power_w() or None,
    }

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--attention', default='sb', choices=['s', 'b', 'sb', 'n'], type=str, metavar="attention",
                    help='attention module')
parser.add_argument('-b', '--backbone', choices=['dino', 'v2', 'resnet'], type=str, metavar="backbone",
                    help='backbone')
parser.add_argument('-o', '--output', type=str, metavar="output",
                    help='output model filename')
parser.add_argument('-e', '--epoch', type=int, metavar="epoch",
                    help='number of epoch')
parser.add_argument('-c', type=int, metavar="class",
                    help='number of class')
parser.add_argument('-l', '--loss', choices=['a', 's', 'as', 'n'], type=str, metavar="loss",
                    help='loss function')
parser.add_argument('-d', '--dataset', choices=['face', 'nose', 'nose_old'], type=str, metavar="dataset",
                    help='what dataset to use')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

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

train_transforms_3 = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.RandomResizedCrop(
        size=(224, 224),
        scale=(0.9, 1.0),
        ratio=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms_4 = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.05
    ),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms_5 = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
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
                    if label >= args.c:
                        continue
                except:
                    continue

                for root, dirs, files in os.walk(class_dir):
                    if os.path.basename(root) != split:
                        continue

                    for fname in files:
                        fpath = os.path.join(root, fname)
                        if os.path.isfile(fpath):
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
    train_dataset_1 = DogDataset('crop_copy', transform=train_transforms_3)
    train_dataset_2 = DogDataset('crop_copy', transform=train_transforms_4)
    train_dataset_3 = DogDataset('crop_copy', transform=train_transforms_5)
    train_dataset_4 = DogDataset('crop_copy', transform=val_transforms)
    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    class DogDataset_eva(Dataset):
        def __init__(self, root_dir, split="train", transform=None, class_num=45):
            self.root_dir = root_dir
            self.split = split
            self.transform = transform
            self.samples = []

            if split == 'unknown':
                unknown_dir = os.path.join(root_dir, 'unknown')
            
                if os.path.isdir(unknown_dir):
                    # Walk specifically inside the 'unknown' folder
                    for root, dirs, files in os.walk(unknown_dir):
                        for fname in natsorted(files):
                            fpath = os.path.join(root, fname)
                            
                            # Filter for valid image extensions
                            if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                                # Assign label -1 for unknown images
                                self.samples.append((fpath, -1))
                
                    # Error handling if empty
                if len(self.samples) == 0:
                    print(f"Warning: No images found in '{unknown_dir}'")
                
                return

            for class_name in sorted(os.listdir(root_dir)):
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue

                try:
                    label = int(class_name)
                    if label < 17:
                        continue
                except:
                    continue

                for root, dirs, files in os.walk(class_dir):
                    if os.path.basename(root) != split:
                        continue

                    for i, fname in enumerate(natsorted(files)):
                        fpath = os.path.join(root, fname)
                        if os.path.isfile(fpath):
                            if split == 'gallery' and i >= 4:
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
        
    val_dataset = DogDataset_eva('crop_copy', 'test', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    val_dataset_unknown = DogDataset_eva('crop_copy', 'unknown', transform=val_transforms, class_num=50)
    val_loader_unknown = DataLoader(val_dataset_unknown, batch_size=16, shuffle=False)

    gallery_dataset = DogDataset_eva('crop_copy', 'gallery', transform=val_transforms)
    gallery_loader = DataLoader(gallery_dataset, batch_size=16,shuffle=False)

## Scheduler
def lr_lambda(epoch):
    if epoch < args.epoch/2:
        return 1.0
    else:
        return max(0.0, 1.0 - (epoch - (args.epoch/2)) / (args.epoch/2))

def build_gallery_centroids(gallery_embeddings, device):
    centroids = {}
    for label, emb_list in gallery_embeddings.items():
        emb_stack = torch.stack(emb_list)
        centroid = emb_stack.mean(dim=0)
        centroids[label] = centroid

    centroid_feats = torch.stack(list(centroids.values())).to(device)
    centroid_feats = F.normalize(centroid_feats, dim=1)
    centroid_labels = torch.tensor(list(centroids.keys())).to(device)

    return centroid_feats, centroid_labels

def evaluate_embedding(model, gallery_loader, val_loader, device, top_k=1):
    model.eval()

    gallery_embeddings = {}
    test_feats = []
    test_labels = []

    with torch.no_grad():

        # ---- Extract Gallery Embeddings ----
        for img, label in gallery_loader:
            img = img.to(device)
            emb = model(img)

            for i in range(len(label)):
                lbl = label[i].item()
                e = emb[i]

                if lbl not in gallery_embeddings:
                    gallery_embeddings[lbl] = [e]
                else:
                    gallery_embeddings[lbl].append(e)

        # ---- Extract Test Embeddings ----
        for img, label in val_loader:
            img = img.to(device)
            emb = model(img)

            test_feats.append(emb)
            test_labels.append(label)

    # Stack test
    test_feats = torch.cat(test_feats, dim=0)
    test_feats = F.normalize(test_feats, dim=1)
    test_labels = torch.cat(test_labels).to(device)

    # Build centroids
    centroid_feats, centroid_labels = build_gallery_centroids(
        gallery_embeddings, device
    )

    # Cosine similarity (vectorized)
    sim = torch.matmul(test_feats, centroid_feats.T)

    topk = torch.topk(sim, k=top_k, dim=1)
    preds = centroid_labels[topk.indices[:, 0]]

    acc = (preds == test_labels).float().mean().item()

    return acc


## Train
def train():
    if args.backbone == 'resnet':
        model = Network_Resnet(args.attention).to(device)
    else:
        model = Network_ConvNext(args.backbone, args.attention).to(device)

    total_params, trainable_params = count_parameters(model)
    model_size_mb = estimate_model_size_mb(model)
    print(f"Model params : {total_params/1e6:.2f}M total, "
          f"{trainable_params/1e6:.2f}M trainable, {model_size_mb:.2f} MB on disk")

    arcface_loss, soft_triple_loss_1, soft_triple_loss_2, soft_triple_loss_3 = None, None, None, None
    if 'a' in args.loss:
        arcface_loss = ArcFace(1024, int(args.c), scale=16, margin=0.6).to(device)
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
    optimizer = AdamW(param_groups, lr=3.5e-5, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    num_epochs = args.epoch

    train_acc_list = []
    val_acc_list = []
    per_epoch_stats = []
    peak_cpu_rss_mb = 0.0
    global_peak_gpu_mb = 0.0

    train_monitor = GPUPowerMonitor(sample_interval=0.5)
    train_monitor.start()
    train_t0 = time.perf_counter()

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

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_t0 = time.perf_counter()

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
                logits_s = logits_s_1 + logits_s_2 + logits_s_3
                combined_logits += torch.softmax(logits_s, dim=1)
            else:
                loss = ce_loss(output, id_set)
                combined_logits = torch.softmax(output, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            preds = torch.argmax(combined_logits, dim=1).to(torch.device("cpu"))
            # print(preds[:10], id_set[:10])
            correct += (preds == dog_id).sum().item()
            total += dog_id.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time_s = time.perf_counter() - epoch_t0
        epoch_peak_gpu_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                             if device.type == "cuda" else None)
        if epoch_peak_gpu_mb is not None and epoch_peak_gpu_mb > global_peak_gpu_mb:
            global_peak_gpu_mb = epoch_peak_gpu_mb
        rss_now = cpu_rss_mb()
        if rss_now is not None and rss_now > peak_cpu_rss_mb:
            peak_cpu_rss_mb = rss_now

        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {training_loss/len(train_loader):.4f}, Train Acc: {train_acc * 100:.4f}")

        val_acc_embed = evaluate_embedding(model, gallery_loader, val_loader, device)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc_embed)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Acc (logits)     : {train_acc*100:.2f}%")
        print(f"Val Acc (embedding NN) : {val_acc_embed*100:.2f}%")
        print(f"Epoch time             : {epoch_time_s:.1f} s"
              + (f"  | peak GPU {epoch_peak_gpu_mb:.0f} MB"
                 if epoch_peak_gpu_mb is not None else "")
              + (f"  | CPU RSS {rss_now:.0f} MB" if rss_now is not None else ""))

        per_epoch_stats.append({
            "epoch": epoch + 1,
            "train_loss": training_loss / len(train_loader),
            "train_acc": train_acc,
            "val_acc_embed": val_acc_embed,
            "epoch_time_s": epoch_time_s,
            "peak_gpu_mem_mb": epoch_peak_gpu_mb,
            "cpu_rss_mb": rss_now,
        })

        scheduler.step()

    total_train_time_s = time.perf_counter() - train_t0
    train_monitor.stop()
    train_energy_j = train_monitor.energy_joules()
    train_avg_power_w = train_monitor.avg_power_w()
    train_peak_power_w = train_monitor.peak_power_w()

    print("\n=== Training compute summary ===")
    print(f"Total training time : {total_train_time_s:.1f} s "
          f"({total_train_time_s/60:.2f} min, "
          f"mean {total_train_time_s/max(num_epochs,1):.1f} s/epoch)")
    if global_peak_gpu_mb:
        print(f"Peak GPU memory     : {global_peak_gpu_mb:.0f} MB")
    if peak_cpu_rss_mb:
        print(f"Peak CPU RSS        : {peak_cpu_rss_mb:.0f} MB")
    if train_energy_j > 0:
        print(f"GPU energy          : {train_energy_j/1000:.2f} kJ "
              f"({train_energy_j/3.6e6:.4f} kWh), "
              f"avg {train_avg_power_w:.1f} W, peak {train_peak_power_w:.1f} W")

    torch.save(model.state_dict(), f"model/model/dino/{args.output}.pt")
    if arcface_loss is not None:
        torch.save(arcface_loss.state_dict(), f"{args.output}_arcface.pt")
    if soft_triple_loss_1 is not None:
        torch.save(soft_triple_loss_1.state_dict(), f"{args.output}_soft_ensemble_1.pt")
        torch.save(soft_triple_loss_2.state_dict(), f"{args.output}_soft_ensemble_2.pt")
        torch.save(soft_triple_loss_3.state_dict(), f"{args.output}_soft_ensemble_3.pt")

    plt.figure()
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.output}_accuracy_curve.png")
    plt.close()

    # Edge-deployment proxy: latency / memory / energy per image on val_loader.
    inference_stats = None
    if "val_loader" in globals():
        print("\nProfiling inference on val_loader for edge-deployment numbers...")
        inference_stats = profile_inference(model, globals()["val_loader"], device)
        if inference_stats["latency_ms_per_image"] is not None:
            print(f"Inference latency   : {inference_stats['latency_ms_per_image']:.2f} ms/image "
                  f"({inference_stats['throughput_imgs_per_s']:.1f} imgs/s)")
        if inference_stats["peak_gpu_mem_mb"] is not None:
            print(f"Peak inference GPU  : {inference_stats['peak_gpu_mem_mb']:.0f} MB")
        if inference_stats["energy_mj_per_image"] is not None:
            print(f"Inference energy    : {inference_stats['energy_mj_per_image']:.2f} mJ/image "
                  f"(avg {inference_stats['avg_gpu_power_w']:.1f} W)")

    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": gpu_static_info(),
        },
        "config": {
            "backbone": args.backbone,
            "attention": args.attention,
            "loss": args.loss,
            "dataset": args.dataset,
            "num_classes": args.c,
            "epochs": num_epochs,
            "output": args.output,
        },
        "model": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "size_mb": model_size_mb,
        },
        "training": {
            "total_time_s": total_train_time_s,
            "total_time_min": total_train_time_s / 60.0,
            "mean_epoch_time_s": total_train_time_s / max(num_epochs, 1),
            "peak_gpu_mem_mb": global_peak_gpu_mb or None,
            "peak_cpu_rss_mb": peak_cpu_rss_mb or None,
            "energy_j": train_energy_j if train_energy_j > 0 else None,
            "energy_kwh": train_energy_j / 3.6e6 if train_energy_j > 0 else None,
            "avg_gpu_power_w": train_avg_power_w if train_energy_j > 0 else None,
            "peak_gpu_power_w": train_peak_power_w if train_energy_j > 0 else None,
            "per_epoch": per_epoch_stats,
        },
        "inference": inference_stats,
    }

    report_base = f"model/model/dino/{args.output}_compute_report"
    with open(report_base + ".json", "w") as f:
        json.dump(report, f, indent=2)

    with open(report_base + ".txt", "w") as f:
        env = report["environment"]
        cfg = report["config"]
        m = report["model"]
        t = report["training"]
        f.write("Computational efficiency report\n")
        f.write("================================\n\n")
        f.write(f"Run    : {cfg['output']}  ({cfg['backbone']}/{cfg['attention']}, "
                f"loss={cfg['loss']}, dataset={cfg['dataset']}, "
                f"classes={cfg['num_classes']}, epochs={cfg['epochs']})\n")
        f.write(f"Python : {env['python']}    Torch: {env['torch']}    "
                f"CUDA: {env['cuda_version']}\n")
        f.write(f"OS     : {env['platform']}\n")
        if env['gpu'].get('name'):
            cap = env['gpu'].get('power_limit_w')
            f.write(f"GPU    : {env['gpu']['name']}  "
                    f"({env['gpu'].get('total_memory_mb', 0):.0f} MB"
                    + (f", {cap:.0f} W cap" if cap else "") + ")\n")
        f.write("\nModel\n-----\n")
        f.write(f"  Parameters : {m['total_params']/1e6:.2f}M total, "
                f"{m['trainable_params']/1e6:.2f}M trainable\n")
        f.write(f"  Size       : {m['size_mb']:.2f} MB\n")
        f.write("\nTraining\n--------\n")
        f.write(f"  Total time   : {t['total_time_s']:.1f} s "
                f"({t['total_time_min']:.2f} min)\n")
        f.write(f"  Mean / epoch : {t['mean_epoch_time_s']:.1f} s\n")
        if t['peak_gpu_mem_mb']:
            f.write(f"  Peak GPU mem : {t['peak_gpu_mem_mb']:.0f} MB\n")
        if t['peak_cpu_rss_mb']:
            f.write(f"  Peak CPU RSS : {t['peak_cpu_rss_mb']:.0f} MB\n")
        if t['energy_j']:
            f.write(f"  GPU energy   : {t['energy_j']/1000:.2f} kJ "
                    f"({t['energy_kwh']:.4f} kWh)\n")
            f.write(f"  GPU power    : avg {t['avg_gpu_power_w']:.1f} W, "
                    f"peak {t['peak_gpu_power_w']:.1f} W\n")
        if inference_stats is not None:
            f.write("\nInference (edge-deployment proxy)\n")
            f.write("---------------------------------\n")
            if inference_stats['latency_ms_per_image'] is not None:
                f.write(f"  Latency      : {inference_stats['latency_ms_per_image']:.2f} ms/image\n")
                f.write(f"  Throughput   : {inference_stats['throughput_imgs_per_s']:.1f} imgs/s\n")
            if inference_stats['peak_gpu_mem_mb'] is not None:
                f.write(f"  Peak GPU mem : {inference_stats['peak_gpu_mem_mb']:.0f} MB\n")
            if inference_stats['energy_mj_per_image'] is not None:
                f.write(f"  Energy       : {inference_stats['energy_mj_per_image']:.2f} mJ/image "
                        f"(avg {inference_stats['avg_gpu_power_w']:.1f} W)\n")

    print(f"\nWrote compute report -> {report_base}.json / .txt")

train()