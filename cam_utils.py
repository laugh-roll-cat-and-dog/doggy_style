import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.handle_forward = target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output

        if hasattr(self.activations, 'requires_grad') and self.activations.requires_grad:
            self.activations.retain_grad()

    def remove_hooks(self):
        self.handle_forward.remove()

    def generate_cam(self, input_image, target_class_idx=None):
        model_output = self.model(input_image)

        self.model.zero_grad()
        score = torch.norm(model_output)

        score.backward()

        if self.activations is None:
            raise ValueError("No activations found. Make sure the target layer is correct.")
        if self.activations.grad is None:
            raise ValueError("No gradients found. Ensure that the model's output is connected to the target layer.")
        
        gradients = self.activations.grad.detach().cpu()
        activations = self.activations.detach().cpu()

        pooled_gradients = torch.mean(gradients, dim=[2, 3])

        activations = activations[0]
        grads = pooled_gradients[0]

        weighted_activations = activations * grads.view(-1, 1, 1)

        heatmap = torch.sum(weighted_activations, dim=0).numpy()

        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else heatmap

        return heatmap
    
def overlay_heatmap(heatmap, original_image_path, alpha=0.5, colormap=cv2.COLORMAP_JET):
    original_image = None
    
    if isinstance(original_image_path, Image.Image):
        original_image_path = np.array(original_image_path)

    if isinstance(original_image_path, str):
        original_image = cv2.imread(original_image_path)

    elif isinstance(original_image_path, np.ndarray):
        original_image = original_image_path.copy().astype(np.uint8)

        if len(original_image.shape) == 3 and original_image.shape[2] == 3: 
            try:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error converting image color: {e}")
    else:
        raise ValueError("original_image_path must be a file path or a numpy array.")   
    
    if original_image is None:
        raise ValueError("Could not read the image from the provided path.")

    original_image = cv2.resize(original_image, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    superimposed_image = heatmap_colored * alpha + original_image
    superimposed_image = np.clip(superimposed_image, 0, 255).astype(np.uint8)

    return superimposed_image, heatmap_colored