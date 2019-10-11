import os
from os import path
import tqdm
import os
import cv2
import glob
import multiprocessing as mp
import numpy as np
import pydicom
import types
from fastai2.medical.imaging   import *

# from ww import f
# Functions
dicom_windows = types.SimpleNamespace(
    brain=(80,40),
    subdural=(200,80),
    stroke=(8,32),
    brain_bone=(2800,600),
    brain_soft=(375,40),
    lungs=(1500,-600),
    mediastinum=(350,50),
    abdomen_soft=(400,50),
    liver=(150,30),
    spine_soft=(250,50),
    spine_bone=(1800,400)
)


def hist_scaled_px(self:DcmDataset, brks=None, min_px=None, max_px=None):
    px = self.scaled_px
    if min_px is not None: px[px<min_px] = min_px
    if max_px is not None: px[px>max_px] = max_px
    return px.hist_scaled()


def convert_to_png_fastai(dcm_in):
    save_path = os.path.join(dir_img, os.path.basename(dcm_in)[:-3] + 'npy')
    if path.exists(save_path):
        return
    dcm = dcmread(dcm_in)
    tensor = dcm.windowed(*dicom_windows.subdural).numpy()
    
    np.save(save_path, tensor)

# Extract images in parallel
p_size = int(mp.cpu_count() / 2)

dir_dcm = '/home/jupyter/rsna/source_data/stage_1_test_images'
dir_img = '/home/jupyter/rsna/rsna-train-stage-1-images_fastai_subdural_windows/stage_1_test_npy'

if not path.exists(dir_img):
    print(f'make dir => {dir_img}')
    os.makedirs(dir_img, )

dicom = glob.glob(os.path.join(dir_dcm, '*.dcm'))

pool = mp.Pool(p_size)
for _ in tqdm.tqdm(pool.imap_unordered(convert_to_png_fastai, dicom), total=len(dicom)):
    pass
pool.close()



dir_dcm = '/home/jupyter/rsna/source_data/stage_1_train_images'
dir_img = '/home/jupyter/rsna/rsna-train-stage-1-images_fastai_subdural_windows/stage_1_train_npy'

if not path.exists(dir_img):
    print(f'make dir => {dir_img}')
    os.makedirs(dir_img, )
    
dicom = glob.glob(os.path.join(dir_dcm, '*.dcm'))

pool = mp.Pool(p_size)
for _ in tqdm.tqdm(pool.imap_unordered(convert_to_png_fastai, dicom), total=len(dicom)):
    pass
pool.close()