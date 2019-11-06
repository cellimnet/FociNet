from tensorflow.keras.applications.vgg19 import VGG19
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from static import VGG_IMG_WIDTH
from static import VGG_IMG_HEIGHT
from static import VGG_IMG_CHANNELS
import os
import cv2 as cv

### Set parameters
test_layer = 'block5_conv4'  #test layer
img_size = 16    #define parameter acoording to（img_size, img_size, img_n）
img_n = 512      #define parameter acoording to（img_size, img_size, img_n）

    ### Parameters for joint
img_jg = 5   #interval between images, default is 5
img_co_num = 4   #how many images in a row
dir_feature = './input/feature/'   #route of test images

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

folder_name_feature_single = './save/feature_single/'
folder_name_feature_joint = './save/feature_joint/'
if not os.path.exists(folder_name_feature_single):
    os.mkdir(folder_name_feature_single)
if not os.path.exists(folder_name_feature_joint):
    os.mkdir(folder_name_feature_joint)

### Main
for name in os.listdir(dir_feature):
    name_out = os.path.splitext(name)[0]
    data = cv.imread(dir_feature + name)
    data = data.reshape(1,256,256,3)

    # Visualize the output of middle layer
    conv_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer(test_layer).output)
    conv_output_raw = conv_layer_model.predict(data)
    conv_output_raw = conv_output_raw.reshape(img_size,img_size,img_n)
    conv_output_raw = np.uint8(255 * conv_output_raw)

    # Output the first 64
    for n in range(64):
        cv.imwrite(folder_name_feature_single + name_out + '_' + test_layer +  '_' + str(n) + 's' + '.tif' , conv_output_raw[:,:,n])
    print('Output of middle layer --single: ' + name)

    ### Joint
    img_sst = 0
    img_whole_size = img_size * img_co_num + img_jg * (img_co_num - 1)
    img_if = img_whole_size - img_size
    img_whole_rgb = np.zeros((img_whole_size, img_whole_size, 3))
    for t in range(int(img_n/(img_co_num*img_co_num))):
        img_rst = 0
        img_whole = np.zeros((img_whole_size, img_whole_size))
        for m in range(img_co_num):
            img_cst = 0
            for n in range(img_sst, img_sst + img_co_num):
                img_whole[img_rst:img_rst + img_size, img_cst:img_cst + img_size] = conv_output_raw[:, :, n]
                if img_cst != img_if:
                    img_cst = img_cst + img_size + img_jg
            img_rst = img_rst + img_size + img_jg
            img_sst = img_sst + img_co_num
        img_whole_rgb[:, :, 1] = img_whole[:, :]
        cv.imwrite(folder_name_feature_joint + name_out + '_' + test_layer + '_' + str(t+1) + '.tif' , img_whole)
    print('Output of middle layer --joint: ' + name)

    ### add pseudo color
    # conv_output_mean = np.mean(conv_output_raw, -1)
    # conv_output_mean = np.uint8(255 * conv_output_mean)
    # conv_output_mean_rgb = np.zeros((img_size, img_size, 3))
    # conv_output_mean_rgb[:,:,1] = conv_output_mean[:,:]
    # cv.imwrite('./save/feature_map/' + name_out + '_' + test_layer + '_mean' + '.tif', conv_output_mean_rgb)
    # for j in range(30):
    #     cv.imwrite('./save/feature_map/' + name_out + '_' + test_layer + '_sub'  + '_' + str(j) + '.tif', conv_output_raw[:,:,j])

