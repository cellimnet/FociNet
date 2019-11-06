from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import os, time
from func_def import cell_count
from static import VGG_IMG_WIDTH

from static import VGG_IMG_HEIGHT
from static import VGG_IMG_CHANNELS
from static import VGG_BATCH_SIZE
from static import dir_single_ec_floder

### Route
test_data_path = dir_single_ec_floder
drug_names = os.listdir(test_data_path)

### Clear files
with open('./save/cell_count.csv','a') as f_count:
    f_count.seek(0)
    f_count.truncate()
    f_count.write('%s,%s,%s,%s\n'%('DRUG_NAME', 'CLASS 0', 'CLASS 1', 'CLASS 2'))
with open('./save/cell_test.csv','a') as f_test:
    f_test.seek(0)
    f_test.truncate()
    f_test.write('%s,%s,%s,%s,%s\n'%('NAME', 'CLASS 0', 'CLASS 1', 'CLASS 2', 'CLASS'))
with open('./save/cell_num.csv','a') as f_num:
    f_num.seek(0)
    f_num.truncate()
    f_num.write('%s,%s\n'%('NAME', 'CLASS'))

### Build VGG
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
trainer.compile(optimizer='adam',#tf.keras.optimizers.Adam(lr=0.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
trainer.load_weights('./asset/VGG.h5')
# classify.summary()

### Get results
test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(VGG_IMG_HEIGHT, VGG_IMG_WIDTH),
        batch_size=VGG_BATCH_SIZE,
        shuffle=False)
print("Start VGGPre")
print("Waiting...")
test_generator.reset()
pred = trainer.predict_generator(
        test_generator)

filenames = test_generator.filenames
pred = np.array(pred)
pred2 = np.zeros((pred.shape[0],))
for i in range(pred.shape[0]):
    pred2[i] = np.argmax(pred[i,:])

### Output csv file
with open('./save/cell_test.csv','a') as file_print:
    for f,p1,p2 in zip(filenames,pred,pred2):
        file_print.write("%s,%lf,%lf,%lf,%d\n"%(f,p1[0],p1[1],p1[2],p2))

with open('./save/cell_num.csv', 'a') as cell_print:
    for f, p1, p2 in zip(filenames, pred, pred2):
        cell_print.write("%s,%d\n" %(f,p2))

with open('./save/cell_num.csv', 'r') as f_num:
    words = f_num.readlines()
    for name in drug_names:
        print("Drug: %s..."%name)
        cell_count(name, words)

print("FINISH VGGPre!")