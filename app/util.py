import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2

def grad_cam(model, input_tensor, target_layer):
    activations = gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    class_idx = output.argmax()
    output[0, class_idx].backward()

    handle_forward.remove()
    handle_backward.remove()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    return heatmap / heatmap.max()

def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)).convert("RGB")
    return Image.blend(img, heatmap_img, alpha=0.5)