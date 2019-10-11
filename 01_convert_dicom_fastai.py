import os
from os import path
import tqdm
import os
import cv2
import glob
import multiprocessing as mp
import numpy as np
import pydicom
from fastai2.medical.imaging   import *

# from ww import f
# Functions

    
def convert_to_png_fastai(dcm_in):
    save_path = os.path.join(dir_img, os.path.basename(dcm_in)[:-3] + 'npy')
    if path.exists(save_path):
        return
    dcm = dcmread(dcm_in)
    tensor = dcm.scaled_px().numpy()
    
    np.save(save_path, tensor)

# Extract images in parallel

dir_dcm = '/home/jupyter/rsna/source_data/stage_1_test_images'
dir_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_test_png_224x_fastai'

if not path.exists(dir_img):
    print(f'make dir => {dir_img}')
    os.makedirs(dir_img, )

dicom = glob.glob(os.path.join(dir_dcm, '*.dcm'))

pool = mp.Pool(mp.cpu_count())
for _ in tqdm.tqdm(pool.imap_unordered(convert_to_png_fastai, dicom), total=len(dicom)):
    pass
pool.close()



dir_dcm = '/home/jupyter/rsna/source_data/stage_1_train_images'
dir_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x_fastai'

if not path.exists(dir_img):
    print(f'make dir => {dir_img}')
    os.makedirs(dir_img, )
    
dicom = glob.glob(os.path.join(dir_dcm, '*.dcm'))

pool = mp.Pool(mp.cpu_count())
for _ in tqdm.tqdm(pool.imap_unordered(convert_to_png_fastai, dicom), total=len(dicom)):
    pass
pool.close()