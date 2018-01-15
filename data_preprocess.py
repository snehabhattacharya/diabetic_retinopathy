import glob
import os
import sys

import numpy as np
import skimage
from skimage import io


root = '/home/sneha_bhattacharya227/diabetic_retinopathy/dataset'
ds_factor = int(sys.argv[1])

def process_images():
    labels = ['symptoms', 'nosymptoms']
    phase = ['train', 'val']
    for p in phase:
        dataset_dir = os.path.join(root, p)
        for l in labels:
            dataset_dir1 = os.path.join(dataset_dir, l)
            print dataset_dir1
            train_files = list(set(glob.glob(os.path.join(dataset_dir1, "*.jpeg"))))
            crop_images(train_files, p)
    

def crop_images(train_files, phase):
    for i, img_id in enumerate(train_files):
        # print img_id

        im = skimage.io.imread(img_id)
        # im = Image.open(img_id, mode='r')

        im_new = im[::ds_factor, ::ds_factor]
        cols_thres = np.where(
            np.max(
                np.max(
                    np.asarray(im_new),
                    axis=2),
                axis=0) > 30)[0]

        if len(cols_thres) > 2:
            min_x, max_x = cols_thres[0], cols_thres[-1]
        else:
            min_x, max_x = 0, -1

        rows_thres = np.where(
            np.max(
                np.max(im_new,
                       axis=2),
                axis=1) > 30)[0]

        if len(rows_thres) > 2:
            min_y, max_y = rows_thres[0], rows_thres[-1]
        else:
            min_y, max_y = 0, -1

        im_new = im_new[min_y:max_y, min_x:max_x]

        w, h = im_new.shape[:2]
        print img_id
        skimage.io.imsave(
            img_id.replace(phase,
                phase +
                # str(ds_factor) +
                '_ds_crop'),
            im_new)

        if i % 1000 == 0:
            print i

process_images()
