import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load your dataset (adjust path as necessary)
data_dir = r'C:\Users\Lenovo\OneDrive\Desktop\dataset\dataset\datasetforclassification\final\test'
image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Load the pre-trained ResNet18 model
model = models.resnet18(weights=None)  # Updated: use weights=None instead of pretrained=False
num_ftrs = model.fc.in_features
num_classes = 8  # Set to 8 as per your information
model.fc = nn.Linear(num_ftrs, num_classes)

# Load your trained weights
model.load_state_dict(torch.load(r"C:\Users\Lenovo\OneDrive\Desktop\dataset\dataset\datasetforclassification\dance_model.pth", map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

# Grad-CAM function
def grad_cam(model, input_tensor, target_layer):
    # Hook the specified layer
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    # Get the model output
    model_output = model(input_tensor)
    
    # Clear existing gradients
    model.zero_grad()
    
    # Backward pass with the output corresponding to the predicted class
    model_output[:, model_output.max(1)[1]].backward()
    
    # Remove the hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Weight the channels by corresponding gradients
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # Average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    
    # ReLU on top of the heatmap
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    
    # Normalize the heatmap
    heatmap = heatmap / np.max(heatmap)
    
    return heatmap

# Function to generate and save Grad-CAM visualizations
def save_gradcam(model, input_tensor, img_path):
    model.eval()
    
    # Get the number of layers in model.layer4
    num_layers = len(list(model.layer4))
    print(f'Number of layers in model.layer4: {num_layers}')
    
    # Select appropriate layers based on the number of layers in model.layer4
    layers = []
    layer_names = []
    if num_layers >= 3:
        layers = [model.layer4[-1], model.layer4[-2], model.layer4[-3]]
        layer_names = ['layer4[-1]', 'layer4[-2]', 'layer4[-3]']
    elif num_layers == 2:
        layers = [model.layer4[-1], model.layer4[-2]]
        layer_names = ['layer4[-1]', 'layer4[-2]']
    elif num_layers == 1:
        layers = [model.layer4[-1]]
        layer_names = ['layer4[-1]']
    
    gradcams = []

    for layer in layers:
        gradcams.append(grad_cam(model, input_tensor, layer))

    # Load original image for overlay
    orig_img = Image.open(img_path).convert('RGB')
    orig_img = orig_img.resize((224, 224))
    
    plt.figure(figsize=(20, 5))
    for i, heatmap in enumerate(gradcams):
        plt.subplot(1, len(gradcams) + 1, i + 1)
        plt.imshow(orig_img)
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.title(f'{layer_names[i]}')
        plt.axis('off')
    
    # Add original image
    plt.subplot(1, len(gradcams) + 1, len(gradcams) + 1)
    plt.imshow(orig_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'gradcam_{os.path.basename(img_path)}')
    plt.close()

# Generate Grad-CAM for a few images
num_images = 5
images_so_far = 0

for img_name in image_files[:num_images]:
    img_path = os.path.join(data_dir, img_name)
    image = Image.open(img_path).convert('RGB')
    input_tensor = data_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    save_gradcam(model, input_tensor, img_path)
    
    images_so_far += 1
    if images_so_far == num_images:
        break

print(f"Grad-CAM visualizations for {num_images} images have been saved.")
