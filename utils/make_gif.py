from dipy.io.image import load_nifti

import numpy as np
import imageio
import matplotlib.pyplot as plt

from monai.transforms import Resize

def overlay_mask_on_image(image, mask, gif_filename, axis=0, colors=None):
    """
    Create a GIF by overlaying a segmentation mask on an image.

    Parameters:
    - image: 3D numpy array of the brain image.
    - mask: 3D numpy array of the mask with labels.
    - gif_filename: str, filename for the output GIF.
    - axis: int, axis along which to create slices.
    - colors: list of RGB values for each label.
    """
    
    # Default colors if not provided
    if colors is None:
        colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Normalize image for visualization
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # Create a blank RGB canvas
    combined_rgb = np.zeros((image.shape[axis], image.shape[1], image.shape[2], 3), dtype=np.uint8)
    
    for i in range(image.shape[axis]):
        slice_img = np.take(image, indices=i, axis=axis)
        slice_mask = np.take(mask, indices=i, axis=axis)

        for label, color in enumerate(colors):
            combined_rgb[i][slice_mask == label] = color
        
        # Blend image with mask (adjust alpha as needed)
        combined_rgb[i] = (0.15 * combined_rgb[i] + 0.85 * slice_img[:, :, None]).astype(np.uint8)

    # Save as GIF
    imageio.mimsave(gif_filename, combined_rgb, duration=0.1)

# Example usage
if __name__ == '__main__':
    # Dummy data (replace with actual data)
    image, _ = load_nifti("/home/fran/DATA/rsna-2023-abdominal-trauma-detection/nifti_files/201_49066.nii.gz")
    mask, _ = load_nifti("/home/fran/Projects/RSNA-2023/utils/201_49066/mask.nii.gz")  # 5 labels including 0

    transform_img = Resize((224, 224, 224))
    transform_mask = Resize((224,224,224), mode="nearest")
    timg = transform_img(
        np.expand_dims(image, axis=0)
    )[0, ...]
    tmask = transform_mask(
        np.expand_dims(mask, axis=0)
    )[0, ...]
    overlay_mask_on_image(np.rot90(timg, k=1, axes=(0, 1)), np.rot90(tmask, k=1, axes=(0, 1)), 'output.gif', axis=2)
