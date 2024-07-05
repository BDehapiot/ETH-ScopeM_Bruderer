#%% Imports -------------------------------------------------------------------

import random
import napari
import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), 'data', 'train') 

# Parameters
random.seed(42)
contrast_limits = (0.1, 1)
brush_size = 5

#%%

metadata = []
for path in train_path.iterdir():
    if "mask" not in path.name:
        metadata.append({
            "name"  : path.name,
            "path"  : path,
            })
  
#%%

# initialize viewer
viewer = napari.Viewer()
viewer.text_overlay.visible = True

# Open first image
while True:
    idx   = random.randint(0, len(metadata) - 1)
    path  = metadata[idx]["path"]
    if not Path(str(path).replace(".tif", "_mask.tif")).exists():
        image = io.imread(path)
        mask  = np.zeros_like(image, dtype="uint8")
        viewer.add_image(image, name="image", metadata=metadata[idx])
        viewer.add_labels(mask, name="mask")
        viewer.layers["image"].contrast_limits = contrast_limits
        viewer.layers["image"].gamma = 0.66
        viewer.layers["mask"].brush_size = brush_size
        viewer.layers["mask"].selected_label = 2
        viewer.layers["mask"].mode = 'paint'
        viewer.text_overlay.text = path.name
        break 
    
def next_image():
    
    # Save previous mask
    path = viewer.layers["image"].metadata["path"]
    path = Path(str(path).replace(".tif", "_mask.tif"))
    io.imsave(path, viewer.layers["mask"].data, check_contrast=False)  
    
    # Open next image
    while True:
        idx   = random.randint(0, len(metadata))
        path  = metadata[idx]["path"]
        if not Path(str(path).replace(".tif", "_mask.tif")).exists():
            image = io.imread(path)
            mask  = np.zeros_like(image, dtype="uint8")
            viewer.layers["image"].data = image
            viewer.layers["image"].metadata = metadata[idx]
            viewer.layers["mask" ].data = mask
            viewer.layers["mask"].selected_label = 2
            viewer.text_overlay.text = path.name
            viewer.reset_view()
            break 
        
def skip_image():
       
    # Open next image
    while True:
        idx   = random.randint(0, len(metadata))
        path  = metadata[idx]["path"]
        if not Path(str(path).replace(".tif", "_mask.tif")).exists():
            image = io.imread(path)
            mask  = np.zeros_like(image, dtype="uint8")
            viewer.layers["image"].data = image
            viewer.layers["image"].metadata = metadata[idx]
            viewer.layers["mask" ].data = mask
            viewer.layers["mask"].selected_label = 2
            viewer.text_overlay.text = path.name
            viewer.reset_view()
            break 

def next_label():
    viewer.layers["mask"].selected_label += 1 

def previous_label():
    if viewer.layers["mask"].selected_label > 2:
        viewer.layers["mask"].selected_label -= 1 
        
def erase():
    viewer.layers["mask"].mode = 'erase'
    
def pan():
    viewer.layers["mask"].mode = 'pan_zoom'
    
def paint():
    viewer.layers["mask"].mode = 'paint'
    
# Shortcut
@napari.Viewer.bind_key('Enter', overwrite=True)
def next_image_key(viewer):
    next_image()
    
@napari.Viewer.bind_key('Shift', overwrite=True)
def skip_image_key(viewer):
    skip_image()
    
@napari.Viewer.bind_key('PageUp', overwrite=True)
def next_label_key(viewer):
    next_label()
    
@napari.Viewer.bind_key('PageDown', overwrite=True)
def previous_label_key(viewer):
    previous_label()
    
@napari.Viewer.bind_key('Backspace', overwrite=True)
def erase_switch(viewer):
    erase()
    yield
    paint()
    
@napari.Viewer.bind_key('Space', overwrite=True)
def pan_switch(viewer):
    pan()
    yield
    paint()

napari.run()    