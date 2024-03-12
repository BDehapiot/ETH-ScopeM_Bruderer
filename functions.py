#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
from skimage.transform import rescale

#%% Functions -----------------------------------------------------------------

def get_voxel_size(file_path):

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        return 1.

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            z = 1.
        tags = tiff.pages[0].tags
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        return [z, y, x]
    
# -----------------------------------------------------------------------------

def format_stack(stack, voxSize, normalize=True):
    
    if normalize:
        pMax = np.percentile(stack, 99.99)
        stack[stack > pMax] = pMax
        stack = (stack / pMax).astype(float)
           
    # Rescale & reslice (isotropic voxel)
    ratio = voxSize[1] / voxSize[0]
    rscale = rescale(stack, (1, ratio, ratio), order=0)
    rslice = np.swapaxes(rscale, 0, 1)
        
    return rscale, rslice