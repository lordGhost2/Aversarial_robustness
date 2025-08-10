import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from medmnist import PathMNIST
import numpy as np
import cv2
from PIL import Image

# -------- SETUP -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("gradcam_outputs", exist_ok=True)
OUTPUT_RES = (1600, 1600)  # Final image size (width, height)

# -------- MODEL DEFINITION -------- #
def get_resnet(num_classes=9):
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# -------- HOOK FOR GRAD-CAM -------- #
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]       # [C, H, W]
        activations = self.activations[0]   # [C, H, W]
        weights = gradients.mean(dim=(1, 2))  # Global average pooling

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        return cam.cpu().numpy()

    def close(self):
        for handle in self.hook_handles:
            handle.remove()

# -------- DATA LOADING -------- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
test_dataset = PathMNIST(split='test', download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# -------- LOAD MODEL -------- #
model = get_resnet().to(device)
model.load_state_dict(torch.load("blackbox_model.pth", map_location=device))

# -------- GRAD-CAM INITIALIZE -------- #
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

# -------- INVERSE NORMALIZATION -------- #
inv_normalize = transforms.Normalize(mean=[-1], std=[2])  # Reverse normalization

# -------- GENERATE & SAVE HIGH-RES VISUALS -------- #
for idx, (img, label) in enumerate(test_loader):
    if idx >= 20:
        break
    img = img.to(device)
    cam = gradcam(img)

    # Convert image from normalized tensor to uint8 RGB
    orig_img = inv_normalize(img.squeeze()).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    orig_img = (orig_img * 255).astype(np.uint8)

    # Resize original image to 1600x1600
    orig_img_resized = cv2.resize(orig_img, OUTPUT_RES, interpolation=cv2.INTER_CUBIC)

    # Generate heatmap from CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, OUTPUT_RES, interpolation=cv2.INTER_CUBIC)

    # Overlay heatmap on image
    overlay = 0.4 * heatmap_resized + 0.6 * orig_img_resized
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Save high-res result
    out_img = Image.fromarray(overlay)
    out_path = f"gradcam_outputs/img_{idx:02d}_class_{label.item()}.png"
    out_img.save(out_path)
    print(f"[✓] Saved: {out_path}")

gradcam.close()
print("\n✅ All Grad-CAM visualizations saved to: gradcam_outputs/")
