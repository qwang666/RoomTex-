from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def show_img(img):
    show_img = Image.fromarray(img)
    plt.figure('image')
    plt.imshow(show_img)
    plt.show()


def show_depth(img):
    # shape: [H, W]
    vmax = np.percentile(img, 100)
    normalizer = mpl.colors.Normalize(vmin=img.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    colormapped_im = (mapper.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colormapped_im)
    plt.figure('image')
    plt.imshow(img)
    plt.show()
