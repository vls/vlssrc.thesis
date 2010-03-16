import ctypes_opencv
from ctypes_opencv import *
from ctypes_opencv.interfaces import *


def binarize(pilimage):
    cvImage = cvCreateImageFromPilImage(pilimage)
    
    cvThreshold(cvImage, cvImage, 0, 255, CV_THRESH_OTSU)
    
    
    
    return cvImage.as_pil_image()

def main():
    print dir(ctypes_opencv)
    

if __name__ == '__main__':
    main()
    
    