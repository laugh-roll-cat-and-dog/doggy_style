import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import numpy as np

from cam_utils import GradCAM, overlay_heatmap
from network.network import Network_Resnet, Network_ConvNext

argparser = argparse.ArgumentParser(description="Visualize CAM for Dog Breed Verification")
argparser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the trained model file')
argparser.add_argument('-i', '--image_path', type=str, required=True, help='Path to the input image')
argparser.add_argument('-a', '--attention', type=str, choices=['s', 'b','sb','d'], default='resnet', help='Model attention to use')
argparser.add_argument('-bb', '--backbone', type=str, choices=['v2', 'dino'], help='backbone features to make residual connection')
argparser.add_argument('-o', '--output_path', type=str, default='cam_output.png', help='Path to save the output image with CAM overlay')
argparser.add_argument('-arc', '--architecture', type=str, choices=['resnet', 'convnext'], default='resnet', help='Architecture of the model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparser.parse_args()
model_path = args.model_path
image_path = args.image_path

def run_visualization():
    if args.architecture == 'resnet':
        model = Network_Resnet(attention=args.attention)
    else:
        model = Network_ConvNext(attention=args.attention, backbone=args.backbone)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True


    # List of target layers for CAM visualization
    targets = {}

    if args.architecture == 'resnet':
        if hasattr(model, 'extra_layers') and hasattr(model, 'backbone'):
            if model.extra_layers:
                targets['1. Before Attention'] = model.extra_layers[-1]
    elif args.architecture == 'convnext':        
        target_found = False
        
        if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'stages'):
            targets['1. Before Attention'] = model.backbone.encoder.stages[-1]
            target_found = True
            
        elif hasattr(model.backbone, 'stages'):
            targets['1. Before Attention'] = model.backbone.stages[-1]
            target_found = True

        if not target_found:
            if hasattr(model.backbone, 'layernorm'):
                    targets['1. Before Attention'] = model.backbone.layernorm
            elif hasattr(model.backbone, 'layer_norm'):
                    targets['1. Before Attention'] = model.backbone.layer_norm
    
    # SAM
    if hasattr(model, 'sam') and model.sam is not None:
        targets['2. After Attention (SAM)'] = model.sam

    # BAM
    if hasattr(model, 'bam') and model.bam is not None:
        targets['3. After Attention (BAM)'] = model.bam
    
    # DAM
    if hasattr(model, 'dam') and model.dam is not None:
        targets['4. After Attention (DAM)'] = model.dam
    
    # Orchestra (Fusion)
    if hasattr(model, 'orchestra') and model.orchestra is not None:
        targets['5. After Orchestra (Fusion)'] = model.orchestra

    # Loop for Grad-CAM visualization
    results = []
    titles = []

    results.append(np.array(raw_image))
    titles.append("Original Image")

    for name, target_layer in targets.items():
        print(f"Generating CAM for: {name}")
        cam = GradCAM(model=model, target_layer=target_layer)
        try:
            heatmap = cam.generate_cam(input_tensor)

            overlay_img, _ = overlay_heatmap(heatmap, args.image_path)

            results.append(overlay_img)

            titles.append(name)
        
        except Exception as e:
            print(f"Could not generate CAM for {name}: {e}")
            continue

        cam.remove_hooks()
        model.zero_grad()

    # Grid
    num_imgs = len(results)
    plt.figure(figsize=(4 * num_imgs, 5))

    for i in range(num_imgs):
        plt.subplot(1, num_imgs, i + 1)
        plt.imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i], fontsize=10)
        plt.axis('off')    

    image_name = args.image_path.replace('\\','/').split('/')[-1]
    image_name = image_name.rstrip('.jpg')
    model_name = model_path.replace('\\','/').split('/')[-1]
    model_name = model_name.rstrip('.pt')

    plt.tight_layout()
    plt.savefig(f"cam_result_{image_name}_{model_name}.png")
    print(f"บันทึกรูปผลลัพธ์ที่: cam_result_{image_name}_{model_name}.png")
    plt.show()

if __name__ == "__main__":
    run_visualization()
