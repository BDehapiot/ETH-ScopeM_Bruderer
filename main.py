#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
from skimage import io
from pathlib import Path

# Functions
from functions import get_voxel_size, format_stack

#%% Inputs --------------------------------------------------------------------

local_path = "D:\local_Bruderer\data"
# stack_name = "Ede1_Magenta_DNA_Green_Example2.tif"
# stack_name = "Sur7_Magenta_DNA_Green_Example1.tif"
stack_name = "Tcb3_Magenta_DNA_Green_Example2.tif"

stack_path = Path(local_path, stack_name)

#%% Process -------------------------------------------------------------------

# Open data & metadata
stack = io.imread(stack_path)
voxSize = get_voxel_size(stack_path)
C1, C2, C3 = stack[..., 0], stack[..., 1], stack[..., 2]
C3 = C3[np.argmax(np.mean(C3, axis=(1, 2))), ...]     

# Get rslice
C1_rscale, C1_rslice = format_stack(C1, voxSize)
C2_rscale, C2_rslice = format_stack(C2, voxSize)

#%% Display -------------------------------------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(C1_rslice, scale=[1, 1, 1])
viewer.add_image(C2_rslice, scale=[1, 1, 1])
# viewer.add_image(C1, scale=voxSize, gamma=0.66, blending="additive", colormap="magenta")
# viewer.add_image(C2, scale=voxSize, gamma=0.66, blending="additive", colormap="green")