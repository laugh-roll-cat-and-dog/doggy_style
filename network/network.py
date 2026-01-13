import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import  ConvNextV2ForImageClassification, AutoModel

from attention.BAM import BAM
from attention.DAM import ChannelAttentionModule, PositionAttentionModule, DualAttentionModule
from attention.SAM import Self_Attention
from attention.SEblock import FeatureFusionModule

class Network_ConvNext(nn.Module):
    def __init__(self, backbone, attention, embedding_dim=1024):
        super().__init__()
        self.attention = attention
        self.bb = backbone
        if backbone == 'dino':
            model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
            model = AutoModel.from_pretrained(
                model_name, 
                device_map="cuda", 
            )
            self.backbone = model
            num_features = model.layer_norm.normalized_shape[0]
        elif backbone == 'v2':
            model_name = "facebook/convnextv2-tiny-1k-224"
            base_model = ConvNextV2ForImageClassification.from_pretrained(model_name)
            self.backbone = base_model.convnextv2
            num_features = base_model.classifier.in_features

        self.cam = ChannelAttentionModule()
        self.pam = PositionAttentionModule(num_features)
        self.dam = DualAttentionModule(in_channels=num_features)
        self.sam = Self_Attention(num_features)
        self.bam = BAM(num_features)
        
        in_chan = 0
        if attention == 'sb':
            in_chan = num_features * 2
        else:
            in_chan = num_features

        self.orchestra = FeatureFusionModule(
            in_chan=in_chan,
            out_chan=3*num_features,
            attention=attention,
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3*num_features, embedding_dim, bias=False)
        self.fc_abl = nn.Linear(num_features, embedding_dim, bias=False)
        self.ln = nn.LayerNorm(embedding_dim)

    def extract_backbone(self, x):
        out = self.backbone(x)
        if self.bb == 'dino':
            feat = out.last_hidden_state
            feat = feat[:, 1:, :]
            B, N, C = feat.shape
            
            H = W = int(N ** 0.5)

            feat = feat.transpose(1, 2).reshape(B, C, H, W)
            return feat
        return out.last_hidden_state

    def embed(self, x):
        feat = self.extract_backbone(x)

        if self.attention == 's':
            fused = self.sam(feat)
        elif self.attention == 'b':
            fused = self.bam(feat)
        elif self.attention == 'sb':
            att1 = self.sam(feat)
            att2 = self.bam(feat)
            fused = self.orchestra(fsp=att1, fcp=att2)
        else:
            fused = feat

        fused = self.gap(fused)
        fused = fused.view(fused.size(0), -1)

        if self.attention == 'sb':
            x = self.fc(fused)
        else:
            x = self.fc_abl(fused)
        x = self.ln(x)
        # x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img):
        return self.embed(img)

class Network_Resnet(nn.Module):
    def __init__(self, attention, embedding_dim=1024):
        super().__init__()
        self.attention = attention
        resnet = models.resnet50()
        resnet.load_state_dict((torch.load("byol_stan.pt", map_location="cuda")))
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
        if self.attention == 'sb':
            in_chan = 1024
        else:
            in_chan = 512
        self.orchestra = FeatureFusionModule(
            in_chan=in_chan,
            out_chan=512,
            attention=attention,
        )

        self.fc = nn.Linear(512, embedding_dim, bias=False)  
        self.bn = nn.BatchNorm1d(embedding_dim)

    def embed(self, x):
        x = self.backbone(x)
        x = self.extra_layers(x)   
        if self.attention == 's':
            x = self.sam(x)
        elif self.attention == 'b':
            x = self.bam(x)
        elif self.attention == 'sb':
            att1 = self.sam(x)
            att2 = self.bam(x)
            x = self.orchestra(fsp=att1, fcp=att2)
        x = self.gap(x)
        x = x.view(x.size(0), -1)      
        x = self.fc(x)             
        x = self.bn(x)

        # x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img):
        emb = self.embed(img)
        return emb