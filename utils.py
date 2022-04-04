import cv2
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import PIL.Image
import random
import torch
import torchvision
from IPython.display import HTML


def display_img(img):
    if type(img) == torch.Tensor:
        fig = plt.figure()
        plt.imshow(img.permute(1,2,0))
    else:
        return PIL.Image.fromarray(img)
    
# Convert list of frames to inline jupyter animation
# Credits to: DM Control Suite tutorial,
# https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb
def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    # matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    # matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    if is_notebook():
        print("displaying in notebook")
        return HTML(anim.to_jshtml())
    else:
        print("displaying in script")
        return anim

# Check if running from notebook or python script
def is_notebook():
    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return True   # notebook
        else:
            return False # script
    except NameError:
        return False
    
# Downscale image to 3x64x64 + transform to normalized tensors
def preprocess_img(img):
    return torchvision.transforms.functional.to_tensor((cv2.resize(img, (64, 64))))