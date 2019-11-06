import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import backend as K

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def enhance_contrast(img_name, save_name):
    img = cv.imread(img_name, 0)
    img_new = img.copy()
    hist= cv.calcHist([img], [0], None, [256], [0,255])
    hrange = [i for i, e in enumerate(hist) if e != 0]
    hmin = hrange[0]
    hmax = hrange[-1]
    hlength = hmax - hmin
    for pix in hrange:
        pix = int(pix)
        if hlength==0:
            hlength=1
        pix_new = round(pix*(255/hlength))
        if int(pix_new) > 255:
            pix_new = 255
        img_new[img == pix] = pix_new
    cv.imwrite(save_name, img_new)

def cell_count(start_word, words):
    all_drug_0 = []
    all_drug_1 = []
    all_drug_2 = []
    drugs_name = []
    with open('./save/cell_count.csv','a') as f_count:
        start_words = start_word + '\\'
        drug_name = start_word
        all_0 = 0
        all_1 = 0
        all_2 = 0
        for word in words:
            word = word.replace(',',' ')
            m,n = word.split()
            if m.startswith(start_words):
                if int(n) == 0:
                    all_0 += 1
                if int(n) == 1:
                    all_1 += 1
                if int(n) == 2:
                    all_2 += 1
        drugs_name.append(drug_name)
        all_drug_0.append(all_0)
        all_drug_1.append(all_1)
        all_drug_2.append(all_2)

        for c1,c2,c3,c4 in zip(drugs_name, all_drug_0, all_drug_1, all_drug_2):
            f_count.write('%s,%s,%s,%s\n'%(c1,c2,c3,c4))

def pyij_ec(img_dir, img_dir_ec):
    from jnius import autoclass
    IJ = autoclass('ij.IJ')
    imp = IJ.openImage(img_dir)
    IJ.run(imp, "Enhance Contrast...", "saturated=0.01")
    IJ.saveAsTiff(imp, img_dir_ec)

def pyij_maskprocess(img_dir, img_dir_ec):
    from jnius import autoclass
    IJ = autoclass('ij.IJ')
    imp = IJ.openImage(img_dir)
    IJ.setAutoThreshold(imp, "Default dark")
    IJ.run(imp, "Analyze Particles...", "  show=Masks exclude include in_situ")
    IJ.saveAsTiff(imp, img_dir_ec)