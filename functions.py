#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
from joblib import Parallel, delayed 
from skimage.transform import rescale
from scipy.ndimage import distance_transform_edt
from skimage.morphology import (
    disk, binary_erosion, binary_dilation
    )

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

# -----------------------------------------------------------------------------

def get_all(msk):
    
    labels = np.unique(msk)[1:]
    edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
    for l, label in enumerate(labels):
        tmp = msk == label
        tmp = distance_transform_edt(tmp)
        pMax = np.percentile(tmp[tmp > 0], 99.9)
        tmp[tmp > pMax] = pMax
        tmp = (tmp / pMax)
        edm[l,...] = tmp
    edm = np.max(edm, axis=0).astype("float32")  
    
    return edm

# -----------------------------------------------------------------------------

def get_outlines(msk):
    
    labels = np.unique(msk)[1:]
    edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
    for l, label in enumerate(labels):
        tmp = msk == label
        tmp = tmp ^ binary_erosion(tmp)
        tmp = binary_dilation(tmp, footprint=disk(1))
        tmp = distance_transform_edt(tmp)
        pMax = np.percentile(tmp[tmp > 0], 99.9)
        tmp[tmp > pMax] = pMax
        tmp = (tmp / pMax)
        edm[l,...] = tmp
    edm = np.max(edm, axis=0).astype("float32")
    
    return edm

# def get_outlines(msk):
    
#     labels = np.unique(msk)[1:]
#     edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
#     for l, label in enumerate(labels):
#         tmp = msk == label
#         tmp = tmp ^ binary_erosion(tmp)
#         edm[l,...] = tmp
#     edm = np.max(edm, axis=0)
#     edm = (pixconn(edm, conn=2) > 3).astype("float32")
#     # edm = binary_dilation(edm, footprint=disk(1))
#     # edm = distance_transform_edt(edm)
#     # pMax = np.percentile(edm[edm > 0], 99.9)
#     # edm[edm > pMax] = pMax
#     # edm = (edm / pMax)
    
#     return edm

# -----------------------------------------------------------------------------

def pixconn(img, conn=2):

    conn1 = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]])
    
    conn2 = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]])
    
    # Convert img as bool
    img = img.astype('bool')
    
    # Pad img with False
    img = np.pad(img, pad_width=1, constant_values=False)
    
    # Find True coordinates
    idx = np.where(img == True) 
    idx_y = idx[0]; idx_x = idx[1]
    
    # Define all kernels
    mesh_range = np.arange(-1, 2)
    mesh_x, mesh_y = np.meshgrid(mesh_range, mesh_range)
    kernel_y = idx_y[:, None, None] + mesh_y
    kernel_x = idx_x[:, None, None] + mesh_x
    
    # Filter image
    all_kernels = img[kernel_y,kernel_x]
    if conn == 1:
        all_kernels = np.sum(all_kernels*conn1, axis=(1, 2))
    if conn == 2:    
        all_kernels = np.sum(all_kernels*conn2, axis=(1, 2))
    img = img.astype('uint8')
    img[idx] = all_kernels
    
    # Unpad img
    img = img[1:-1,1:-1]
    
    return img

# -----------------------------------------------------------------------------

def get_patches(arr, size, overlap):
    
    # Get dimensions
    if arr.ndim == 2: nT = 1; nY, nX = arr.shape 
    if arr.ndim == 3: nT, nY, nX = arr.shape
    
    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2
    
    # Pad array
    if arr.ndim == 2:
        arr_pad = np.pad(
            arr, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect') 
    if arr.ndim == 3:
        arr_pad = np.pad(
            arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2)), mode='reflect')         
    
    # Extract patches
    patches = []
    if arr.ndim == 2:
        for y0 in y0s:
            for x0 in x0s:
                patches.append(arr_pad[y0:y0 + size, x0:x0 + size])
    if arr.ndim == 3:
        for t in range(nT):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size])
            
    return patches

# -----------------------------------------------------------------------------

def merge_patches(patches, shape, size, overlap):
    
    # Get dimensions 
    if len(shape) == 2: nT = 1; nY, nX = shape
    if len(shape) == 3: nT, nY, nX = shape
    nPatch = len(patches) // nT

    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    # Merge patches
    def _merge_patches(patches):
        count = 0
        arr = np.full((2, nY + yPad, nX + xPad), np.nan)
        for i, y0 in enumerate(y0s):
            for j, x0 in enumerate(x0s):
                if i % 2 == j % 2:
                    arr[0, y0:y0 + size, x0:x0 + size] = patches[count]
                else:
                    arr[1, y0:y0 + size, x0:x0 + size] = patches[count]
                count += 1 
        arr = np.nanmean(arr, axis=0)
        arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
        return arr
        
    if len(shape) == 2:
        arr = _merge_patches(patches)

    if len(shape) == 3:
        patches = np.stack(patches).reshape(nT, nPatch, size, size)
        arr = Parallel(n_jobs=-1)(
            delayed(_merge_patches)(patches[t,...])
            for t in range(nT)
            )
        arr = np.stack(arr)
        
    return arr