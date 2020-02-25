### Grad-CAM wrapper for PyTorch (WIP)
Grad-CAM wrapper is a utility that easily allows you extract and visualize gradient-weighted class activation mappings for your custom model. It works by wrapping around your model, without overwriting any of your function definitions.

### Features
* Computes Grad-CAM using your custom model
* Support for multitarget classification
* Grad-CAM computation is differentiable, allowing you to make it trainable
* Multi-GPU support
* Visualization using `opencv`, `matplotlib` or `Pillow`, depending on your choice
* Sensible defaults, but many configurable options
* Licensed under MIT

### Installation
Requirements:
```
python >= 3.6
torch
```

### Examples

### FAQ 
* Can I use it with any of the `torchvision` models?  
Yes, but you cannot use the functions that usually define the model directly, as they are not of the `torch.nn.Module` class. So this requires a little more involvement than usual. An example would be:
```
model = gradcam_wrapper(torchvision.models.ResNet)
model_params = {'block': torchvision.models.resnet.BasicBlock, 'layers': [2, 2, 2, 2]}
gradcam_layer = 'layer4'
model = model(model_params, gradcam_layer)
```
Refer to the PyTorch source to do the same for other models

### References
