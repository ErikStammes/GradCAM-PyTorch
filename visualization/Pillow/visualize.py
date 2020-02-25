import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision.transforms import transforms

def save_gradcam(filename, gcam, raw_image):
    gcam = gcam.detach().cpu().numpy()
    cmap = plt.cm.jet_r(gcam)[..., :3] * 255.0
    cmap = cmap[:, :, ::-1]
    cmap = Image.fromarray(cmap.astype(np.uint8))
    raw_image = raw_image.permute(1, 2, 0).cpu().numpy() * 255
    transform = transforms.ToPILImage()
    raw_image = transform(raw_image.astype(np.uint8))
    result = Image.blend(raw_image, cmap, 0.5)
    result.save(filename)
