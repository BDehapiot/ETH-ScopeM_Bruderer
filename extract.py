#%% Imports -------------------------------------------------------------------

import random 
import numpy as np
from skimage import io
from pathlib import Path

# Functions
from functions import get_voxel_size, format_stack

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Bruderer/data')
train_path = Path(Path.cwd(), 'data', 'train') 

# Parameters
random.seed(42)
nImg = 5

#%% Extract -------------------------------------------------------------------

for path in list(local_path.glob("*.tif")):
        
    # Open data & metadata
    hstack = io.imread(path)
    voxSize = get_voxel_size(path)
    C1, C2, C3 = hstack[..., 0], hstack[..., 1], hstack[..., 2]
    C3 = C3[np.argmax(np.mean(C3, axis=(1, 2))), ...]     
    
    # Format stacks
    C1C2 = C1 + C2
    C1C2_rscale, C1C2_rslice = format_stack(C1C2, voxSize)
    
    # Random indexes
    idx_rscale = np.random.choice(
        range(C1C2_rscale.shape[0]), size=nImg, replace=False)
    idx_rslice = np.random.choice(
        range(C1C2_rslice.shape[0]), size=nImg, replace=False)
    
    # Save images
    for idx in idx_rscale:
        img_name = path.name.replace(".tif", f"_rscale_{idx:03d}.tif")
        io.imsave(
            Path(train_path, img_name),
            C1C2_rscale[idx, ...].astype("float32"), check_contrast=False,
            )    
    for idx in idx_rslice:
        img_name = path.name.replace(".tif", f"_rslice_{idx:03d}.tif")
        io.imsave(
            Path(train_path, img_name),
            C1C2_rslice[idx, ...].astype("float32"), check_contrast=False,
            )    
    