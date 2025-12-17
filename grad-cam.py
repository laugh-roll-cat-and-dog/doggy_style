import torch
import cv2
import numpy as np
import argparse
from network.network import Network
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, metavar="model",
                    help=' model filename')
parser.add_argument('-a', '--attention', default='d', choices=['d', 'sb', 'dsb', 'n'], type=str, metavar="attention",
                    help='attention module')
parser.add_argument('-bf', choices=['t', 'f'], type=str, metavar="bf",
                    help='concat backbone feature maps to last feature maps')
args = parser.parse_args()

class EmbeddingTarget:
    def __call__(self, model_output):
        return (model_output ** 2).sum() 

class SimilarityTarget:
    def __init__(self, target_embedding):
        self.target_embedding = target_embedding.detach()
    
    def __call__(self, model_output):
        return torch.sum(model_output * self.target_embedding)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network(args.attention, args.bf).to(device)
# model.load_state_dict(torch.load(f"../model/convnextv2/3_d_orchestra/{args.model}.pt", map_location=device))
model.load_state_dict(torch.load(f"{args.model}.pt", map_location=device))
model.eval()
print(model)

target_layers = [model.orchestra.convblk.conv_bn_relu[0]]

cam = GradCAM(model=model, target_layers=target_layers)

img_path_valid = "../dog/pet_biometric_challenge_2022/train/images/A_ZVriR7xLaJEAAAAAAAAAAAAAAQAAAQ.jpg"
# img_path_valid = "../dogFaces/BG_Brownies_7.jpg"
rgb_img_valid = cv2.imread(img_path_valid, 1)[:, :, ::-1]
rgb_img_valid = cv2.resize(rgb_img_valid, (224, 224))
rgb_img_valid = np.float32(rgb_img_valid) / 255
input_tensor_valid = preprocess_image(rgb_img_valid,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]).to(device)

img_path_gallery = "../dog/pet_biometric_challenge_2022/train/images/A_wgjpSbOp8kcAAAAAAAAAAAAAAQAAAQ.jpg"
# img_path_gallery = "../dogFaces/BG_Brownies_2.jpg"
rgb_img_gallery = cv2.imread(img_path_gallery, 1)[:, :, ::-1]
rgb_img_gallery = cv2.resize(rgb_img_gallery, (224, 224))
rgb_img_gallery = np.float32(rgb_img_gallery) / 255
input_tensor_gallery = preprocess_image(rgb_img_gallery,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]).to(device)

with torch.no_grad():
    embedding_B = model(input_tensor_gallery)

targets = [SimilarityTarget(embedding_B)]

for param in model.backbone.parameters():
    param.requires_grad = True

grayscale_cam = cam(input_tensor=input_tensor_valid, targets=targets)

grayscale_cam = grayscale_cam[0, :]

visualization = show_cam_on_image(rgb_img_valid, grayscale_cam, use_rgb=True)

cv2.imwrite(f'{args.model}.jpg', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))