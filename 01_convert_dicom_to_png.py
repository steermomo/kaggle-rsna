import os
from os import path
import tqdm
import os
import cv2
import glob
import multiprocessing as mp
import numpy as np
import pydicom

# from ww import f
# Functions

def window_image(img, window_center, window_width, intercept, slope):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    return img


def get_first_of_dicom_field_as_int(x):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)
    
    
def get_windowing(data):
    """
    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
    """
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def convert_to_png(dcm_in):
    save_path = os.path.join(dir_img, os.path.basename(dcm_in)[:-3] + 'png')
    if path.exists(save_path):
        return
    dcm = pydicom.dcmread(dcm_in)
    window_center, window_width, intercept, slope = get_windowing(dcm)
    
    window_center = 80
    window_width = 200
    
    try:
        img = pydicom.read_file(dcm_in).pixel_array
    except ValueError:
        return
    img = window_image(img, window_center, window_width, intercept, slope)
    cv2.imwrite(save_path, img)

# Extract images in parallel

dir_dcm = '/home/jupyter/rsna/source_data/stage_1_test_images'
dir_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_test_png_224x'

if not path.exists(dir_img):
    print(f'make dir => {dir_img}')
    os.makedirs(dir_img, )

dicom = glob.glob(os.path.join(dir_dcm, '*.dcm'))

pool = mp.Pool(mp.cpu_count())
for _ in tqdm.tqdm(pool.imap_unordered(convert_to_png, dicom), total=len(dicom)):
    pass
pool.close()



dir_dcm = '/home/jupyter/rsna/source_data/stage_1_train_images'
dir_img = '/home/jupyter/rsna/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'

if not path.exists(dir_img):
    print(f'make dir => {dir_img}')
    os.makedirs(dir_img, )
    
dicom = glob.glob(os.path.join(dir_dcm, '*.dcm'))

pool = mp.Pool(mp.cpu_count())
for _ in tqdm.tqdm(pool.imap_unordered(convert_to_png, dicom), total=len(dicom)):
    pass
pool.close()