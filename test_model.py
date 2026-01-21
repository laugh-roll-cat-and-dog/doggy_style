import torch
from torchinfo import summary
from network.network import Network_ConvNext, Network_Resnet

device = torch.device('cpu')
# model = Network_ConvNext(attention='sb',backbone='v2')
# model = Network_ConvNext(attention='sb',backbone='dino')
model = Network_Resnet(attention='sb')

summary(model, input_size=(1, 3, 224, 224), device='cpu')