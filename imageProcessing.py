import os
import numpy as np 
from pydicom import dcmread 
import pydicom
from PIL import Image

def readImage(image, description, extension=""):

    im = pydicom.dcmread(image)

    im = im.pixel_array.astype(float) 

    rescaled_image = ((np.maximum(im, 0))/im.max())*255

    final_image = np.uint8(rescaled_image)

    final_image = Image.fromarray(final_image)

    basewidth = 128
    final_image = final_image.resize((basewidth,basewidth), Image.ANTIALIAS)
    
    extension = extension[:-4] + ".jpg"
    final_image.save("./{}/{}".format(description, extension))

m = 0
description = "TrainingSet128"
for root, dirs, files in os.walk("siim/dicom-images-train"):
    path = root.split(os.sep)
    for file in files:
        
        print("IMAGE #{}".format(m))
        pathName = './'
        for i in range(len(path)):
            pathName += (path[i] + '/')
        pathName += file
        abspath = os.path.abspath(pathName)
        readImage(abspath, description, extension=file)
        m += 1

# m = 0 
# description = "testingSet"
# for root, dirs, files in os.walk("dicom-images-test"):
#     path = root.split(os.sep)
#     for file in files:
#         print("IMAGE #{}".format(m))
#         pathName = './'
#         for i in range(len(path)):
#             pathName += (path[i] + '/')
#         pathName += file
#         abspath = os.path.abspath(pathName)
#         readImage(abspath,  file, description)
#         m += 1
        