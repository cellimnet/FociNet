from tensorflow.keras.applications.vgg19 import VGG19
import numpy as np
from skimage import measure
from math import floor, ceil
import imageio
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from static import VGG_IMG_WIDTH
from static import VGG_IMG_HEIGHT
from static import VGG_IMG_CHANNELS
from func_def import pyij_ec
import os
import cv2 as cv
import imagej
ij = imagej.init('sc.fiji:fiji:2.0.0-pre-10')

### Computation graph
base_model = VGG19(include_top=False,weights='imagenet',input_shape=(VGG_IMG_HEIGHT,VGG_IMG_WIDTH,VGG_IMG_CHANNELS))
xinput = Input(shape=(8,8,512,))
def LMean(input):
    h = K.reshape(input,(-1,64,512))
    return K.mean(h,1)
x = Lambda(LMean)(xinput)
x = Dense(4096, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
y = Dense(3, activation='softmax')(x)
classify = Model(inputs=xinput, outputs=y)
predictions = classify(base_model.output)
trainer = Model(inputs=base_model.input, outputs=predictions)
trainer.load_weights('./asset/VGG.h5')

### Set parameters
test_layer = 'block5_conv4'
dir_test_raw = './input/heatmap/raw/'   #route of test images
dir_test_mask = './input/heatmap/mask/'   #route of masks
dir_save_heatmap_joint ='./save/heatmap_joint/'
dir_save_heatmap_single = './save/heatmap_single/'
dir_save_heatmap_singletemp = './save/heatmap_singletemp/'
dir_save_heatmap_singleec = './save/heatmap_singleec/'

if not os.path.exists(dir_save_heatmap_joint):
    os.mkdir(dir_save_heatmap_joint)
if not os.path.exists(dir_save_heatmap_single):
    os.mkdir(dir_save_heatmap_single)
if not os.path.exists(dir_save_heatmap_singletemp):
    os.mkdir(dir_save_heatmap_singletemp)
if not os.path.exists(dir_save_heatmap_singleec):
    os.mkdir(dir_save_heatmap_singleec)

img_n = 512
def addcolor(data, name, count):
        ### Output heatmap
        testhot = trainer.output
        last_conv_layer = trainer.get_layer(test_layer)
        grads = K.gradients(testhot, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0,1,2))
        iterate = K.function([base_model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([data])
        for i in range(img_n):
            conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        heatmap = cv.resize(heatmap, (VGG_IMG_HEIGHT, VGG_IMG_WIDTH))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        superimposed_img = heatmap + data.reshape(VGG_IMG_HEIGHT, VGG_IMG_WIDTH, VGG_IMG_CHANNELS)
        return superimposed_img

img_size=256
sta=256

for name in os.listdir(dir_test_mask):
    cell_count = 1
    imgs=[]
    frames=[]
    dir_mask = dir_test_mask + name
    raw_name = name.split("_", 1)[1].split(".")[0]
    raw_type = name.split("_", 1)[1].split(".")[1]
    print("start " + raw_name)
    im_raw = cv.imread(dir_test_raw + raw_name + '.' + raw_type)
    im_mask = cv.imread(dir_mask, 0)
    im_sizea, im_sizeb = im_mask.shape
    labels = measure.label(im_mask, connectivity=1)
    props = measure.regionprops(labels)
    labelmax = labels.max() + 1
    im_raw_copy = im_raw.copy()
    num_temp = 0
    name_split = os.path.splitext(name)[0]
    for i in range(1, labelmax):
        if np.sum(labels == i) > 0 and np.sum(labels == i) < 80000:
            im_1 = im_mask.copy()
            im_1[labels != i] = 0
            im_1[labels == i] = 1
            bbox = props[i - 1]['bbox']
            im_1 = im_1.reshape(im_sizea, im_sizeb, 1)
            imgs.append(im_1[bbox[0]:bbox[2], bbox[1]:bbox[3]] * im_raw[bbox[0]:bbox[2], bbox[1]:bbox[3]])
            shape = imgs[num_temp].shape
            shape_x = shape[0]
            shape_y = shape[1]
            add_x = sta - shape_x
            add_y = sta - shape_y
            add_x_l = int(floor(add_x / 2))
            add_x_r = int(ceil(add_x / 2))
            add_y_l = int(floor(add_y / 2))
            add_y_r = int(ceil(add_y / 2))
            if add_x > 0 and add_y > 0:
                imgs[num_temp] = np.pad(imgs[num_temp], ((add_x_l, add_x_r), (add_y_l, add_y_r), (0, 0)), 'constant',
                                        constant_values=(0, 0))
                pre_name = dir_save_heatmap_singletemp + name_split + '_' + str(num_temp+1) + '.tif'
                save_name = dir_save_heatmap_singleec + name_split + '_' + str(num_temp+1) + '.tif'
                cv.imwrite(pre_name, imgs[num_temp])
                pyij_ec(pre_name, save_name)
                imgs[num_temp] = cv.imread(save_name)
                img_new = imgs[num_temp].copy()
                hist = cv.calcHist([imgs[num_temp]], [0], None, [256], [0, 255])
                hrange = [i for i, e in enumerate(hist) if e != 0]
                hmin = hrange[0]
                hmax = hrange[-1]
                hlength = hmax - hmin
                for pix in hrange:
                    pix = int(pix)
                    if hlength == 0:
                        hlength = 1
                    pix_new = round(pix * (255 / hlength))
                    if int(pix_new) > 255:
                        pix_new = 255
                    img_new[imgs[num_temp] == pix] = pix_new
                img_new = img_new.reshape(1, 256, 256, 3)
                img_color=addcolor(img_new, raw_name, num_temp)
                img_color_a, img_color_b, img_color_c = img_color.shape
                img_color_mask=np.zeros((img_color_a, img_color_b))
                img_color_mask=img_color_mask.reshape(img_color_a, img_color_b, 1)
                img_color_mask[add_x_l : add_x_l + shape_x, add_y_l : add_y_l + shape_y]=im_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                img_color_mask_3D=np.dstack((img_color_mask, img_color_mask, img_color_mask))
                img_extract=img_color*img_color_mask_3D
                img_color_mask=img_color_mask.reshape(img_color_a, img_color_b)
                img_color_temp = img_extract[add_x_l : add_x_l + shape_x, add_y_l : add_y_l + shape_y, :]
                im_raw[labels == i]=img_extract[img_color_mask == 1]
                num_temp += 1
                frames.append(cv.cvtColor(im_raw, cv.COLOR_BGR2RGB))
                cv.imwrite(dir_save_heatmap_single + raw_name + str(i) + '.tif', img_color) #Output heatmap_single
    print(len(frames))
    gif_name=dir_save_heatmap_joint + raw_name + ".gif"
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.08)
    cv.imwrite(dir_save_heatmap_joint+ raw_name + '.tiff', im_raw)

print('Output heatmap')

