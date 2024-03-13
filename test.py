#%% Imports -------------------------------------------------------------------

import napari
import random
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# Functions
from functions import pixconn, get_all, get_outlines, get_patches

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), 'data', 'train')
train_type = "rscale" # "rscale" or "rslice"
train_mode = "outlines" # "all" or "outlines"

# Patches
size = 128
overlap = size // 4

#%% Pre-processing ------------------------------------------------------------

imgs, msks = [], []
for path in train_path.iterdir():
    if train_type in path.name and 'mask' in path.name:
        msks.append(io.imread(path))
        imgs.append(io.imread(str(path).replace('_mask', '')))    
msks = np.stack(msks)            
imgs = np.stack(imgs)

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(msks)
# viewer.add_image(imgs) 

#%% ---------------------------------------------------------------------------

from scipy.ndimage import distance_transform_edt
from skimage.morphology import (
    disk, binary_erosion, binary_dilation
    )

# -----------------------------------------------------------------------------

idx = 0
msk = msks[idx]
img = imgs[idx]

def get_outlines(msk):
    
    labels = np.unique(msk)[1:]
    edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
    for l, label in enumerate(labels):
        tmp = msk == label
        tmp = tmp ^ binary_erosion(tmp)
        edm[l,...] = tmp
    edm = np.max(edm, axis=0)
    edm = pixconn(edm, conn=2) > 3
    # edm = binary_dilation(edm, footprint=disk(1))
    # edm = distance_transform_edt(edm)
    # pMax = np.percentile(edm[edm > 0], 99.9)
    # edm[edm > pMax] = pMax
    # edm = (edm / pMax)
    
    return edm

edm = get_outlines(msk)

# Display 
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_image(edm)

