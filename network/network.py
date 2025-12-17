import torch.nn as nn
import torch.nn.functional as F
from transformers import  ConvNextV2ForImageClassification

from attention.BAM import BAM
from attention.DAM import ChannelAttentionModule, PositionAttentionModule, DualAttentionModule
from attention.SAM import Self_Attention
from attention.SEblock import FeatureFusionModule

class Network(nn.Module):
    def __init__(self, attention, bf, embedding_dim=1024):
        super().__init__()
        self.attention = attention
        self.bf = bf
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
        if attention == 'sb' or attention == 'd':
            in_chan = num_features * 2
        elif attention == 'dsb':
            in_chan = num_features * 4
        else:
            in_chan = num_features

        if bf == 't':
            in_chan += num_features

        self.orchestra = FeatureFusionModule(
            in_chan=in_chan,
            out_chan=2 * num_features,
            attention=attention,
            bf=bf
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

        if self.attention == 'd':
            att1 = self.cam(feat)
            att2 = self.pam(feat)
            fused = self.orchestra(cam=att1, pam=att2, x=feat)
        elif self.attention == 'sb':
            att1 = self.sam(feat)
            att2 = self.bam(feat)
            fused = self.orchestra(fsp=att1, fcp=att2, x=feat)
        elif self.attention == 'dsb':
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