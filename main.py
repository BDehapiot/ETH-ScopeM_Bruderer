#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm

# Skimage
from skimage.measure import label
from skimage.filters import gaussian
from skimage.segmentation import expand_labels

# Functions
from functions import get_voxel_size, format_stack, get_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = "D:\local_Bruderer\data"
# hstack_name = "Ede1_Magenta_DNA_Green_Example3.tif"
# hstack_name = "Sur7_Magenta_DNA_Green_Example3.tif"
hstack_name = "Tcb3_Magenta_DNA_Green_Example2.tif"
hstack_path = Path(local_path, hstack_name)

# Model paths
for model_path in Path.cwd().iterdir():
    if "rscale_all_model_weights" in model_path.name: 
        model_rscale_all_path = model_path
    if "rscale_outlines_model_weights" in model_path.name: 
        model_rscale_out_path = model_path
    if "rslice_all_model_weights" in model_path.name: 
        model_rslice_all_path = model_path
    if "rslice_outlines_model_weights" in model_path.name: 
        model_rslice_out_path = model_path
        
# Patches
size = int(model_rscale_all_path.stem.split("_")[-1].split("-")[0])
overlap = int(model_rscale_all_path.stem.split("_")[-1].split("-")[1])

#%% Preprocessing -------------------------------------------------------------

# Open data & metadata
hstack = io.imread(hstack_path)
voxSize = get_voxel_size(hstack_path)
C1, C2, C3 = hstack[..., 0], hstack[..., 1], hstack[..., 2]
C3 = C3[np.argmax(np.mean(C3, axis=(1, 2))), ...]     

# Format stacks
C1C2 = C1 + C2
C1C2_rscale, C1C2_rslice = format_stack(C1C2, voxSize)

# Get patches
patches_rscale = get_patches(C1C2_rscale, size, overlap)
patches_rscale = np.stack(patches_rscale)
patches_rslice = get_patches(C1C2_rslice, size, overlap)
patches_rslice = np.stack(patches_rslice)

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(patches_rscale)
# viewer.add_image(patches_rslice) 

#%% Predict -------------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )

# -----------------------------------------------------------------------------

# Load weights & predict
model.load_weights(model_rscale_all_path) 
predRscaleAll = model.predict(patches_rscale).squeeze()
predRscaleAll = merge_patches(predRscaleAll, C1C2_rscale.shape, size, overlap)
# ---
model.load_weights(model_rscale_out_path) 
predRscaleOut = model.predict(patches_rscale).squeeze()
predRscaleOut = merge_patches(predRscaleOut, C1C2_rscale.shape, size, overlap)
# ---
model.load_weights(model_rslice_all_path) 
predRsliceAll = model.predict(patches_rslice).squeeze()
predRsliceAll = merge_patches(predRsliceAll, C1C2_rslice.shape, size, overlap)
# ---
model.load_weights(model_rslice_out_path) 
predRsliceOut = model.predict(patches_rslice).squeeze()
predRsliceOut = merge_patches(predRsliceOut, C1C2_rslice.shape, size, overlap)

# # Display
# viewer = napari.Viewer()
# viewer.add_image(C1C2_rscale)
# viewer.add_image(predRscaleAll, blending="additive", colormap="bop orange")
# viewer.add_image(predRscaleOut, blending="additive", colormap="bop blue")
# viewer = napari.Viewer()
# viewer.add_image(C1C2_rslice)
# viewer.add_image(predRsliceAll, blending="additive", colormap="bop orange") 
# viewer.add_image(predRsliceOut, blending="additive", colormap="bop blue") 

#%% Postprocessing ------------------------------------------------------------

# Parameters
sigma = (0.5, 1, 1)
out_coeff = 3
thresh = 0.25

# Merge predictions
predRsliceAll = np.swapaxes(predRsliceAll, 0, 1)
predAll = (predRscaleAll + predRsliceAll) / 2
predAll = gaussian(predAll, sigma=sigma)
predRsliceOut = np.swapaxes(predRsliceOut, 0, 1)
predOut = (predRscaleOut + predRsliceOut) / 2
predOut = gaussian(predOut, sigma=sigma)
predictions = predAll - predOut * out_coeff
predictions[predictions < 0.01] = 0

# Make labels
labels = label(predictions > thresh)
labels = expand_labels(labels, distance=5)
labels[predAll < thresh] = 0

# # Display
# viewer = napari.Viewer()
# viewer.add_image(C1C2_rscale)
# viewer.add_labels(labels)
# viewer.add_image(predictions, blending="additive", colormap="bop orange")
# viewer.add_image(predAll, blending="additive", colormap="bop orange")
# viewer.add_image(predOut, blending="additive", colormap="bop blue")

#%% Postprocessing ------------------------------------------------------------

from skimage.transform import resize
labels_rsize = resize(labels, C1.shape, order=0)

# Display
viewer = napari.Viewer()
viewer.add_image(C1, scale=voxSize)
viewer.add_image(C2, scale=voxSize)
viewer.add_labels(labels_rsize, scale=voxSize)