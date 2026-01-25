import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import numpy as np

from cam_utils import GradCAM, overlay_heatmap
from network.network import Network_Resnet, Network_ConvNext, Network_ConvNext_test

argparser = argparse.ArgumentParser(description="Visualize CAM for Dog Breed Verification")
# argparser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the trained model file')
argparser.add_argument('-i', '--image_path', type=str, required=True, help='Path to the input image')
argparser.add_argument('-a', '--attention', type=str, choices=['s', 'b','sb','d','none'], default='none', help='Model attention to use')
argparser.add_argument('-bb', '--backbone', type=str, choices=['v2', 'dino'], help='backbone features to make residual connection')
argparser.add_argument('-o', '--output_path', type=str, default='cam_output.png', help='Path to save the output image with CAM overlay')
argparser.add_argument('-arc', '--architecture', type=str, choices=['resnet', 'convnext'], default='resnet', help='Architecture of the model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparser.parse_args()
# model_path = args.model_path
image_path = args.image_path

model_config = {
    'none' : {'attention': 'none', "model_path": '../test_model/dino_ablation_none.pt'},
    's'    : {'attention': 's', "model_path": '../test_model/dino_ablation_sam.pt'},
    'b'    : {'attention': 'b', "model_path": '../test_model/dino_ablation_bam.pt'},
    'sb'   : {'attention': 'sb', "model_path": '../test_model/dino_main.pt'},
}
def run_visualization():
    loaded_models = {}
    if args.architecture == 'resnet':
        model = Network_Resnet(attention=args.attention)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        for model_name, config in model_config.items():
            print(f"Loading model for configuration: model using attention : {model_name}")
            if model_name == 'sb':
                model = Network_ConvNext_test(attention=config['attention'],backbone=args.backbone)
            else:
                model = Network_ConvNext(attention=config['attention'],backbone=args.backbone)

            model.load_state_dict(torch.load(config['model_path'], map_location=device))
            model.to(device)
            model.eval()
            loaded_models[model_name] = model
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
        model_none = loaded_models['none']
        if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'stages'):
            target_layer = model_none.backbone.encoder.stages[-1]
            target_found = True
            
        elif hasattr(model.backbone, 'stages'):
            target_layer = model_none.backbone.stages[-1]
            target_found = True

        if not target_found:
            if hasattr(model.backbone, 'layernorm'):
                    target_layer = model_none.backbone.layernorm
            elif hasattr(model.backbone, 'layer_norm'):
                    target_layer = model_none.backbone.layer_norm
        if target_layer:
            targets['1. Before Attention'] = (model_none, target_layer)
    # SAM
    model_s = loaded_models['s']
    if hasattr(model, 'sam') and model.sam is not None:
        targets['2. After Attention (SAM)'] = (model_s, model_s.sam)

    # BAM
    model_b = loaded_models['b']
    if hasattr(model, 'bam') and model.bam is not None:
        targets['3. After Attention (BAM)'] = (model_b, model_b.bam)

    # Orchestra (Fusion)
    model_sb = loaded_models['sb']
    if hasattr(model, 'orchestra') and model.orchestra is not None:
        targets['5. After Orchestra (Fusion)'] = (model_sb, model_sb.orchestra)
    # Loop for Grad-CAM visualization
    results = []
    titles = []

    results.append(np.array(raw_image))
    titles.append("Original Image")

    for name, (model, target_layer) in targets.items():
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

    plt.tight_layout()
    plt.savefig(f"cam_result.png")
    print(f"บันทึกรูปผลลัพธ์ที่: cam_result.png")
    plt.show()

if __name__ == "__main__":
    run_visualization()
