#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

local_path = "D:\local_Bruderer\data"
stack_name = "Ede1_Magenta_DNA_Green_Example1.tif"

#%% Process -------------------------------------------------------------------

# Open data & metadata
stack = io.imread(Path(local_path, stack_name))
C1, C2, C3 = stack[..., 0], stack[..., 1], stack[..., 2]
C3 = C3[np.argmax(np.mean(C3, axis=(1, 2))), ...]


#%% Display -------------------------------------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(C1)