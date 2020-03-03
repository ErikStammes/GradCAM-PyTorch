import cv2
import matplotlib.pyplot as plt
import numpy as np

def save_gradcam(filename, gcam, image):
    gcam = gcam.squeeze().detach().cpu().numpy()
    cmap = plt.cm.jet_r(gcam)[..., :3] * 255.0
    image = image.permute(1, 2, 0).cpu().numpy() * 255
    image = image[:, :, ::-1]
    gcam = (cmap.astype(np.float) + image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))
