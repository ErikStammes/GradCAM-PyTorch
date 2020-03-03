## Grad-CAM wrapper for PyTorch (WIP)
Grad-CAM wrapper is a utility that easily allows you extract and visualize gradient-weighted class activation mappings for your custom model. It works by wrapping around your model, without overwriting any of your models properties/function definitions.

### Features
* Computes Grad-CAM using your custom model
* Support for multitarget classification
* Grad-CAM computation is differentiable, allowing you to make it trainable
* Visualization using `opencv`, or `Pillow`, depending on your preference
* Licensed under MIT

### Installation
Minimum requirements:
```
python >= 3.6
torch
```
For visualization:
```
matplotlib
numpy
python-opencv OR Pillow
```

### Example

```diff
+ from gradcam_wrapper import gradcam_wrapper

...

- model = MyCustomModel(params)
+ model = gradcam_wrapper(MyCustomModel)
+ gradcam_layer = 'my_selected_layer'
+ model = model(params, gradcam_layer)

...

for inputs, target in dataloader:
-   logits = model(inputs)
+   logits, gcams = model(inputs)
    # When using the ground truth labels:
+   logits, gcams = model(inputs, labels=targets)
```

### FAQ 
* Can I use it with any of the `torchvision` models?  
Yes, but you cannot use the functions that usually define the model directly, as they are not instances of the `torch.nn.Module` class. So this requires a little more involvement than usual. An example would be:
```python
model = gradcam_wrapper(torchvision.models.ResNet)
model_params = {'block': torchvision.models.resnet.BasicBlock, 'layers': [2, 2, 2, 2]}
gradcam_layer = 'layer4'
model = model(model_params, gradcam_layer)
```
Refer to the PyTorch source to do the same for other models

### References  
The paper that introduces Grad-CAM:
```
@inproceedings{selvaraju2017grad,
  title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={618--626},
  year={2017}
}
```