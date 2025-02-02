# FociNet-Pipeline-for-53BP1

## **Description**

We have built a FociNet pipeline to analyze fluorescent images with intranuclear foci formation to evaluate DNA damage. It can produce accurate nuclei segmentation and subsequent classification of single nuclei. We have tested it on computers with common GPU, and it performs well. The classification part is based on tensorflow 1.12.0.

## **Install**

Operating system: Windows was tested.

Programing language: Python.

Hardware: >= 8G memory, equipped with a CPU with Core i5 or above.

Environment: Python --3.6.5，CUDA --9.1.83， GPU：Nvidia 1080

This project uses Numpy, Opencv, skimage, tqdm, tensorflow, pyimagej. Go check them out if you don't have them, you can install them with conda or pip. 

```python
pip install numpy==1.17.4
pip install opencv-python
pip install scikit-image==0.16.2
pip install tqdm
pip install tensorflow-gpu==1.12.0
conda install pyimagej
```

## **How to use**

Firstly, unzip "VGG.zip" and put "VGG.h5" into the "./asset/" folder.

**Input**

Put your images into corresponding folders before running UNet-VGG.py file. 8-bit TIF format is recommended. Put the nuclei-channel images into "./input/back/", and the foci-channel images into "./input/pre/". Images of the same site should have the same file name. If you have various treatments to analyze and compare, please put images of one treatment into one subfolder like the example dataset. Even if you have only one treatment, please also build a subfolder for it. 

**Normalization of image size (optional)**

The resizing index for PE Operetta CLS (40×, binning = 1) is defined as 1. For example dataset captured with PE Operetta CLS (40×, binning = 1) , there is no need to change the code. For images captured by other imaging platforms or images captured with an altered binning/magnification, three alterations should be made to make the code adaptive to different settings.
1. Give a name to the set of images captured under a certain settings such as 'settings1'. Change the following code in UNet_pre.py. 

  ```python
  image_type = 'pe'    ####Replace 'pe' with 'settings1'
  ```

2. Exclude single-nucleus images of abnormal size according to the following instructions. Calculate the resizing index by (camera pixel size * binning) / (magnification * 0.2963). Calculate the minimum value as 800 / resizing index / resizing index. Calculate the maximum value as 8000 / resizing index / resizing index. Change the following code in UNet_pre.py. 

  ```python
  if image_type == 'pe':    ####Replace 'pe' with 'settings1'
  sel_sizea = 0    ####Replace '0' with the minimum value as calculated
  sel_sizeb = 80000    ####Replace '80000' with the maximum value as calculated
  ```

3. Change the following code in UNet_pre.py. 

  ```python
  if image_type == 'pe':    ####Replace 'pe' with 'settings1'
  shape_x = int(1 * shape_x)    ####Replace '1' with the value of resizing index as calculated
  shape_y = int(1 * shape_y)    ####Replace '1' with the value of resizing index as calculated
  ```

**Output**

Run the UNet-VGG.py file. After running was finished, you will totally get 4 csv files in "./save/". (1) "cell_seg.csv" provides information about how many single nuclei you have got from each image after segmentation. Predicted masks will be in "./save/ mask/". You will also get raw single-nuclei images in "./save/single/" and pre-processed single-nuclei images in "./save/single_ec/". (2)"cell_test.csv" provides information about the probabilities of a single-nucleus image belonging to the three classes, and the last column gives the most possible class that the image belongs to. There are three classes: “0” means “normal”, “1”means “damaged”, and “2” means “pointless”. (3)"cell_num.csv" is  a simple version of "cell_test.csv". (4)"cell_count.csv" provides information about the statistical results of each treatment. We recommend that 0 / (0+1) is calculated to evaluate and compare the DNA damage of different treatments.

**Example dataset**

We have an example dataset. It includes 9 full-field images. The run time is about 3 minutes.

## **Declaration**

The architecture of U-Net is on the basis of https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277/notebook?scriptVersionId=2164855/notebook.

Training of U-Net uses previously published datasets. These datasets are available at https://github.com/kamalkraj/DATA-SCIENCE-BOWL-2018
and https://data.broadinstitute.org/bbbc/BBBC039/,
respectively.
