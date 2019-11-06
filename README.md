# FociNet-Pipeline-for-53BP1

## **Description**

We have built a FociNet pipeline to analyze fluorescent images with intranuclear foci formation to evaluate DNA damage. It can produce accurate nuclei segmentation and subsequent classification of single nuclei. We have tested it on computers with common GPU, and it performs well. The classification part is based on tensorflow 1.12.0.

## **Install**

Operating system: Windows was tested.

Programing language: Python.

Hardware: >= 8G memory, equipped with a CPU with Core i5 or above.

Environment: Python --3.6.5

This project uses Numpy, Opencv, skimage, tqdm, tensorflow,pyimagej,jnius. Go check them out if you don't have them, you can install them with conda or pip. It will take less than 10 minutes to install them. 

```python
conda install numpy
conda install opencv-python
conda install scikit-image
conda install tqdm
conda install tensorflow
conda install pyimagej
pip install jnius
```

## **How to use**

What you need to do is to put your images into corresponding folders, and run UNet-VGG.py file. 8-bit TIF format is recommended.

Put the nuclei-channel images into "./input/back/", and the foci-channel images into "./input/pre/". If you have various treatments to analyze and compare, please put images of one treatment into one subfolder, like the following picture shows. Even if you have only one treatment, please also build a subfolder for it. We provide a sample for you.

After running was finished, you will totally get 4 csv files in "./save/".

"cell_seg.csv" provides information about how many single nuclei you have got from each image after segmentation. Predicted masks will be in "./save/ mask/". You will also get raw single-nuclei images in "./save/single/" and pre-processed single-nuclei images in "./save/single_ec/".

"cell_test.csv" provides information about the probabilities of a single-nucleus image belonging to the three classes, and the last column gives the most possible class that the image belongs to. There are three classes: “0” means “normal”, “1”means “damaged”, and “2” means “nonsense”.

"cell_num.csv" is just a simple version of "cell_test.csv".

"cell_count.csv" provides information about the statistical results of each treatment. We recommend that 0 / (0+1) value is calculated to evaluate and compare the DNA damage of different treatment.

We have an example dataset. It includes 9 full-field images. The run time is about 3 minute