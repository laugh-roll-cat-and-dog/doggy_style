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

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--attention', default='d', choices=['d', 'sb', 'dsb'], type=str, metavar="attention",
                    help='attention module')
parser.add_argument('-bf', choices=['t', 'f'], type=str, metavar="bf",
                    help='concat backbone feature maps to last feature maps')
parser.add_argument('-c', type=int, metavar="class",
                    help='number of class')
parser.add_argument('-m', '--model', type=str, metavar="model",
                    help=' model filename')
parser.add_argument('-ah', '--arcface', type=str, metavar="arcface",
                    help='arcface classifier head filename')
parser.add_argument('-sh', '--softtriple', type=str, metavar="softtriple",
                    help='softtriple classifier head filename')
parser.add_argument('-o', '--output', type=str, metavar="output",
                    help='output csv filename')
parser.add_argument('-l', '--loss', choices=['a', 's', 'as'], type=str, metavar="loss",
                    help='loss function')
args = parser.parse_args()

## Data
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

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = DogDataset(csv_file='test_split_mixed_sorted.csv', transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

gallery_dataset = DogDataset(csv_file='train_split_sampled.csv', transform=val_transforms)
gallery_loader = DataLoader(gallery_dataset, batch_size=16,shuffle=False)

## NN
#---DAM---
class PositionAttentionModule(nn.Module):
    ''' self-attention '''

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along spatial dimensions
        """

        N, C, H, W = x.shape
        query = self.query_conv(x).view(
            N, -1, H*W).permute(0, 2, 1)  # (N, H*W, C')
        key = self.key_conv(x).view(N, -1, H*W)  # (N, C', H*W)

        # caluculate correlation
        energy = torch.bmm(query, key)    # (N, H*W, H*W)
        # spatial normalize
        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, H*W)    # (N, C, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out

class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, H, W = x.shape
        query = x.view(N, C, -1)    # (N, C, H*W)
        key = x.view(N, C, -1).permute(0, 2, 1)    # (N, H*W, C)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, C, C)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out

class DualAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule()

    def forward(self, x):
        pam_out = self.pam(x)  
        cam_out = self.cam(x)  

        out = torch.cat([cam_out, pam_out, x], dim=1)
        return out

#---SAM---
class Self_Attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=None):
        super(Self_Attention,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out  #,attention

#---bam---
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        #self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        f_ch_att = self.channel_att(in_tensor)
        f_spar_att = self.spatial_att(in_tensor)
        f_att = 1 + torch.sigmoid( f_ch_att * f_spar_att )
        #print("shape channel_att/spatial_att: ", self.channel_att(in_tensor).shape,\
        #    self.spatial_att(in_tensor).shape)
        #print("att shape: ", att.shape)
        output_refined_feature = f_att * in_tensor
        #channel_attention = self.channel_att
        #spartial_attention = self.spatial_att
        return output_refined_feature   #, f_att, f_ch_att, f_spar_att
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

#---AA SE block (1)---
class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,
                      False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True))

    def forward(self, x):
        return self.conv_bn_relu(x)

#---AA SE block (2) Cont'd---
class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()
        self.init_weight()

    def forward(self, fsp, fcp, cam, pam, x):
        if args.attention == 'dsb':
            fcat = torch.cat([fsp, fcp, cam, pam], dim=1)
        else:
            fcat = torch.cat([fsp, fcp], dim=1)
        if args.bf == 't':
            fcat = torch.cat([fcat, x], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        #atten = self.softmax(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

#---Final NN---
class Network(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        resnet = models.resnet50()
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
        self.orchesta = FeatureFusionModule(in_chan, 3*512)

        self.fc = nn.Linear(3*512, embedding_dim, bias=False)  
        self.bn = nn.BatchNorm1d(embedding_dim)

    def embed(self, x):
        x = self.backbone(x)
        x = self.extra_layers(x)   
        if args.attention == 'd':
            x = self.dam(x)                 
            x = self.gap(x)
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

## Loss
class ArcFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=16, margin=0.1, easy_margin=False, **kwargs):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).
        
        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is modified from forward1, which takes
        52.49644996365532 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080 Ti.
        """
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        pos = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        sin_theta = torch.sqrt((1.0 - torch.pow(pos, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = pos * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, pos)
        else:
            phi = torch.where(pos > self.th, phi, pos - self.mm)
        # one_hot = torch.zeros(cos_theta.size(), device='cuda')
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        # output = cos_theta + one_hot
        output *= self.scale
        return output

class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc * self.gamma, dim=2)
        simClass = torch.sum(prob * simStruc, dim=2)

        return self.la * simClass

    def loss(self, logits, target):
        marginM = torch.zeros_like(logits).to(logits.device)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin

        lossClassify = F.cross_entropy(self.la*((logits/self.la)-marginM), target)

        if self.tau > 0 and self.K > 1:
            centers = F.normalize(self.fc, p=2, dim=0)
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify

def loopcheck(test_embeddings, gallery_embeddings):
    results = []  # จะเก็บเป็น list ของ dictionary

    for test_label, test_emb_list in test_embeddings.items():
        for idx, test_emb in enumerate(test_emb_list):  # วนทุก test embedding ทีละตัว
            row_result = {"test_label": test_label, "test_index": idx}
            
            sims_dict = {}
            for gallery_label, gallery_emb_list in gallery_embeddings.items():
                sims = []
                for gallery_emb in gallery_emb_list:
                    sim = F.cosine_similarity(test_emb.unsqueeze(0), gallery_emb.unsqueeze(0)).item()
                    sims.append(sim)

                avg_sim = sum(sims) / len(sims) if sims else 0
                sims_dict[gallery_label] = avg_sim  # เก็บค่าเฉลี่ย similarity

            # เพิ่มค่าเฉลี่ยเข้า row_result
            row_result.update(sims_dict)

            # หา gallery label ที่มีค่าเฉลี่ยสูงสุด
            if sims_dict:
                row_result['ans'] = max(sims_dict, key=sims_dict.get)
            else:
                row_result['ans'] = None

            results.append(row_result)
    return results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()

arcface, softtriple = None, None
if 'a' in args.loss:
    arcface = ArcFace(1024, int(args.c)).to(device)
    arcface.load_state_dict(torch.load(args.arcface, map_location=device))
    arcface.eval()

if 's' in args.loss:
    softtriple = SoftTriple(20, 0.1, 0.2, 0.01, 1024, int(args.c), 3).to(device)
    softtriple.load_state_dict(torch.load(args.softtriple, map_location=device))
    softtriple.eval()

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

        if softtriple is not None:
            logits_s = softtriple(emb)
            combined_logits += torch.softmax(logits_s, dim=1)

        if arcface is not None and softtriple is not None:
            combined_logits /= 2.0

        pred_prob = torch.softmax(combined_logits, dim=1)
        top5_prob, top5_idx = torch.topk(pred_prob, k=5, dim=1)

        for i in range(len(img)):
            result.append([
                dog_id[i].item(),
                top5_idx[i][0].item(),
                top5_idx[i][1].item(),
                top5_idx[i][2].item(),
                top5_idx[i][3].item(),
                top5_idx[i][4].item()
            ])
    checking_results = loopcheck(test_embeddings, gallery_embeddings)
    acc_count = 0
    for i, res in enumerate(checking_results):
        if res['test_label'] == res['ans']:
            acc_count += 1
        # print(f"label_1: {res['test_label']}, label_2: {result[i][0]}")
        result[i].append(res['ans'])
    # accuracy = (acc_count / (len(checking_results))*100) if checking_results else 0
    # print(f"Identification Accuracy: {accuracy:.2f}%")

pred_df = pd.DataFrame(result, columns=[
    'dog_id', 'top1', 'top2', 'top3', 'top4', 'top5', 'emb'
])

pred_df.to_csv(f"{args.output}.csv")

y_true = pred_df["dog_id"].values
y_pred_top1 = pred_df["top1"].values
y_pred_emb = pred_df["emb"].values

acc_top1 = accuracy_score(y_true, y_pred_top1)
acc_emb = accuracy_score(y_true, y_pred_emb)
print("Classifier head Accuracy:", acc_top1 * 100)
print("embedding Accuracy:", acc_emb * 100)
