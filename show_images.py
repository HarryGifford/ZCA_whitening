'''
Function to show an m x n matrix of m images of size w x h x c where n = w * h * c.
'''

import torch
import matplotlib.pyplot as plt

def show_images(images: torch.Tensor, w: int, h: int, c: int, filename: str=None) -> None:
    '''
    Function to show an m x n matrix of m images of size w x h x c where n = w * h * c.

    c can be 1 for grayscale images or 3 for RGB images.
    '''
    m = images.size(0)

    # Rescale images to [0, 1].
    images = (images - images.min()) / (images.max() - images.min())

    # Subplot of size sqrt(m) x sqrt(m).
    msqrt = int(m ** 0.5)
    n_images_width = msqrt if msqrt * msqrt == m else msqrt + 1
    n_images_height = msqrt
    _, axs = plt.subplots(n_images_height, n_images_width)
    for i in range(n_images_height):
        for j in range(n_images_width):
            if i * n_images_width + j < m:
                if c == 1:
                    # Reshape the image to w x h.
                    # Don't allow imshow to rescale the image.
                    axs[i, j].imshow(images[i * n_images_width + j].reshape(w, h), cmap='gray', vmin=0, vmax=1)
                else:
                    # Reshape the image to w x h x c.
                    axs[i, j].imshow(images[i * n_images_width + j].reshape(c, w, h).permute(1, 2, 0), vmin=0, vmax=1)
                axs[i, j].axis('off')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def show_image(image: torch.Tensor) -> None:
    '''
    Function to show a single image.
    '''
    plt.imshow(image, cmap='jet')
    plt.axis('off')
    plt.show()