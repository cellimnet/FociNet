import os, warnings, random
from math import floor, ceil
from skimage import measure
from skimage import morphology, feature
import numpy as np
import cv2 as cv
import skimage.morphology as sm
from skimage.transform import resize
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from func_def import pyij_ec
from func_def import pyij_maskprocess
from func_def import enhance_contrast
from func_def import mean_iou
from static import dir_back_floder
from static import dir_deenv_floder
from static import dir_mask_floder
from static import dir_maskdeenv_floder
from static import dir_maskprocess_floder
from static import dir_pre_floder
from static import dir_single_ec_floder
from static import dir_single_floder
from static import UNet_IMG_WIDTH
from static import UNet_IMG_HEIGHT
from static import UNet_IMG_CHANNELS
from static import UNet_SINGLE_SIZE
import imagej
ij = imagej.init('sc.fiji:fiji:2.0.0-pre-10')

### Choose image type
image_type = 'pe'

### Create folders
drug_names = os.listdir(dir_back_floder)
folder_names = [dir_deenv_floder, dir_mask_floder, dir_maskdeenv_floder, dir_maskprocess_floder, dir_single_floder, dir_single_ec_floder]
for folder_name in folder_names:
    for drug_name in drug_names:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        if not os.path.exists(folder_name + drug_name):
            os.mkdir(folder_name + drug_name)

### Build Unet
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

inputs = Input((UNet_IMG_HEIGHT, UNet_IMG_WIDTH, UNet_IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

### Load parameters
model_deenv = Model(inputs=[inputs], outputs=[outputs])
model_deenv.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model_deenv = load_model('./asset/Unet_deenv.h5', custom_objects={'mean_iou': mean_iou})

model_seg = Model(inputs=[inputs], outputs=[outputs])
model_seg.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model_seg = load_model('./asset/Unet_seg.h5', custom_objects={'mean_iou': mean_iou})

### Def unet_pre
def unet_pre(drug_name, sta):
    dir_back = dir_back_floder + drug_name + "/"
    dir_pre = dir_pre_floder + drug_name + "/"
    dir_deenv = dir_deenv_floder + drug_name + "/"
    dir_mask = dir_mask_floder + drug_name + "/"
    dir_maskprocess = dir_maskprocess_floder + drug_name + "/"
    dir_maskdeenv = dir_maskdeenv_floder + drug_name + "/"
    dir_single = dir_single_floder + drug_name + "/"
    dir_single_ec = dir_single_ec_floder + drug_name + "/"
    print("Drug: %s"%(drug_name))
    count = 1
    with open('./save/cell_seg.csv','a') as f_seg:
        for name_back, name_pre in zip(os.listdir(dir_back), os.listdir(dir_pre)):
            img_deenv = cv.imread(dir_back + name_back)
            img_sizea = img_deenv.shape[0]
            img_sizeb = img_deenv.shape[1]
            ### if a!=b then crop
            if img_sizea != img_sizeb:
                if img_sizea > img_sizeb:
                    crop_a = floor(img_sizea/2) - floor(img_sizeb/2)
                    img_deenv = img_deenv[crop_a: crop_a+img_sizeb, :, :]
                else:
                    crop_b = floor(img_sizeb / 2) - floor(img_sizea / 2)
                    img_deenv = img_deenv[:, crop_b: crop_b + img_sizea, :]
            img_deenv = resize(img_deenv, (UNet_IMG_HEIGHT, UNet_IMG_WIDTH), mode='constant', preserve_range=True)
            X_test_deenv = img_deenv.reshape(1, UNet_IMG_HEIGHT, UNet_IMG_WIDTH, UNet_IMG_CHANNELS)
            preds_test_deenv = model_deenv.predict(X_test_deenv, verbose=1)
            preds_test_deenv = (preds_test_deenv > 0.5).astype(np.uint8)
            pre_img_deenv = np.uint8(np.squeeze(preds_test_deenv[0]) * 255)
            if img_sizea > img_sizeb:
                pre_img_deenv = resize(pre_img_deenv, (img_sizeb, img_sizeb), mode='constant', preserve_range=True)
            else:
                pre_img_deenv = resize(pre_img_deenv, (img_sizea, img_sizea), mode='constant', preserve_range=True)
            cv.imwrite(dir_maskdeenv + "Mask_" + name_pre, pre_img_deenv)
            ####
            pre_img_deenv = cv.imread(dir_maskdeenv + "Mask_" + name_pre, 0)
            img_raw = cv.imread(dir_back + name_back)
            if img_sizea > img_sizeb:
                crop_a = floor(img_sizea/2) - floor(img_sizeb/2)
                img_raw = img_raw[crop_a: crop_a+img_sizeb, :, :]
                pre_img_deenv = pre_img_deenv.reshape(img_sizeb, img_sizeb, 1)
            elif img_sizea < img_sizeb:
                crop_b = floor(img_sizeb / 2) - floor(img_sizea / 2)
                img_raw = img_raw[:, crop_b: crop_b + img_sizea, :]
                pre_img_deenv = pre_img_deenv.reshape(img_sizea, img_sizea, 1)
            else:
                pre_img_deenv = pre_img_deenv.reshape(img_sizea, img_sizeb, 1)
            pre_img_deenv = pre_img_deenv / 255
            pre_img_deenv = img_raw * pre_img_deenv
            cv.imwrite(dir_deenv + "Deenv_" + name_pre, pre_img_deenv)
            ###
            pre_img_deenv = cv.imread(dir_deenv + "Deenv_" + name_pre, 0)
            img_seg = resize(pre_img_deenv, (UNet_IMG_HEIGHT, UNet_IMG_WIDTH), mode='constant', preserve_range=True)
            X_test_seg = img_seg.reshape(1, 512, 512, 1)
            preds_test_seg = model_seg.predict(X_test_seg, verbose=1)
            preds_test_seg = (preds_test_seg > 0.5).astype(np.uint8)
            pre_img_seg = np.uint8(np.squeeze(preds_test_seg[0]) * 255)
            if img_sizea > img_sizeb:
                pre_img_seg = resize(pre_img_seg, (img_sizeb, img_sizeb), mode='constant', preserve_range=True)
            else:
                pre_img_seg = resize(pre_img_seg, (img_sizea, img_sizea), mode='constant', preserve_range=True)
            pre_img_seg[pre_img_seg > 40] = 255
            pre_img_seg = sm.opening(pre_img_seg, sm.disk(9))
            ###
            dir_maskprocessname = dir_maskprocess + "Mask_" + name_pre
            dir_mask_name = dir_mask + "Mask_" + name_pre
            cv.imwrite(dir_maskprocessname, pre_img_seg)
            pyij_maskprocess(dir_maskprocessname, dir_mask_name)
            pre_img_seg = cv.imread(dir_mask_name, 0)
            img_pre = cv.imread(dir_pre + name_pre)
            if img_sizea != img_sizeb:
                if img_sizea > img_sizeb:
                    crop_a = floor(img_sizea/2) - floor(img_sizeb/2)
                    img_pre = img_pre[crop_a: crop_a+img_sizeb, :, :]
                else:
                    crop_b = floor(img_sizeb / 2) - floor(img_sizea / 2)
                    img_pre = img_pre[:, crop_b: crop_b + img_sizea, :]
            name_split = os.path.splitext(name_pre)[0]
            labels = measure.label(pre_img_seg, connectivity=1)
            props = measure.regionprops(labels)
            labelmax = labels.max() + 1
            imgs = []
            num_temp = 0
            sel_sizea = 0
            sel_sizeb = 80000
            if image_type == 'pe':
                sel_sizea = 0
                sel_sizeb = 80000
            for i in range(1, labelmax):
                if np.sum(labels == i) > sel_sizea and np.sum(labels == i) < sel_sizeb:
                    im_1 = pre_img_seg.copy()
                    im_1[labels != i] = 0
                    im_1[labels == i] = 1
                    if img_sizea > img_sizeb:
                        im_1 = im_1.reshape(img_sizeb, img_sizeb, 1)
                    else:
                        im_1 = im_1.reshape(img_sizea, img_sizea, 1)
                    bbox = props[i - 1]['bbox']
                    imgs.append(im_1[bbox[0]:bbox[2], bbox[1]:bbox[3]] * img_pre[bbox[0]:bbox[2], bbox[1]:bbox[3]])
                    shape = imgs[num_temp].shape
                    shape_x = shape[0]
                    shape_y = shape[1]
                    if image_type == 'pe':
                        shape_x = int(1 * shape_x)
                        shape_y = int(1 * shape_y)
                        imgs[num_temp] = resize(imgs[num_temp], (shape_x, shape_y), mode='constant',
                                             preserve_range=True)
                    add_x = sta - shape_x
                    add_y = sta - shape_y
                    add_x_l = int(floor(add_x / 2))
                    add_x_r = int(ceil(add_x / 2))
                    add_y_l = int(floor(add_y / 2))
                    add_y_r = int(ceil(add_y / 2))
                    if add_x > 0 and add_y > 0:
                        imgs[num_temp] = np.pad(imgs[num_temp], ((add_x_l, add_x_r), (add_y_l, add_y_r), (0, 0)), 'constant',
                                                constant_values=(0, 0))

                        pre_name = dir_single + name_split + '_' + str(num_temp + 1) + ".tif"
                        save_name = dir_single_ec + name_split + '_' + str(num_temp + 1) + ".tif"
                        cv.imwrite(pre_name, imgs[num_temp])
                        if image_type == 'confocal':
                            enhance_contrast(pre_name, save_name)
                        else:
                            pyij_ec(pre_name, save_name)
                        num_temp += 1
            print('Figure: ' + str(count))
            print('GOT: ' + str(num_temp))
            print()
            f_seg.write('%s,%d\n' % (drug_name + "_" + name_pre, num_temp))
            count += 1

### Main
with open('./save/cell_seg.csv','a') as f_seg:
        f_seg.seek(0)
        f_seg.truncate()
        f_seg.write('%s,%s\n'%('NAME', 'GOT'))
for drug_name in drug_names:
    unet_pre(drug_name, UNet_SINGLE_SIZE)
print("FINISH UnetPre!")
print()