import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # Placeholders for gradients and activations
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _save_gradients(self, grad):
        self.gradients = grad

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            output.register_hook(self._save_gradients)

        self.target_layer.register_forward_hook(forward_hook)

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)  # Forward pass
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2))  # Global average pooling over time
        cam = (weights.unsqueeze(-1) * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-5)
        return cam.squeeze().cpu().numpy()