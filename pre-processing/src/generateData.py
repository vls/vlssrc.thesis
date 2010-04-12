# -*- coding: utf-8 -*-
from __future__ import with_statement
from PIL import Image
import sys
import struct
import mycv
import picHandler
import copy

TRAIN_LABEL = 'data/The MNIST database of handwritten digits/train-labels.idx1-ubyte'
TRAIN_IMAGE = 'data/The MNIST database of handwritten digits/train-images.idx3-ubyte'

class HImage():
    def __init__(self):
        self.label = -1
        #self.row = 0
        #self.col = 0
        #self.arr = []
        self.image = None
    
    def __deepcopy__(self, memo):
        new = HImage()
        new.label = self.label
        new.image = self.image.copy()
        return new
        
def writeBMP(himage, filename):

    
    
    temphimage = copy.deepcopy(himage)
    
    afterImage = processImage(temphimage).image
    
    thin = picHandler.thinning(temphimage.image)
    thin.save(filename + '_thin.bmp', 'BMP')
    
    afterImage.save(filename + '_trim.bmp', 'BMP')
    
    #print list(after.getdata())
    himage.image.save(filename + '.bmp', 'BMP')

def processImage(himage):
    
    pilImage = himage.image
    oriSize = (pilImage.size[0], pilImage.size[1])
    factor = 3.0 / 4
    newSize = (int(pilImage.size[0] * factor), int(pilImage.size[1] *factor))
    #pilImage = pilImage.resize(newSize, Image.ANTIALIAS) 
    pilImage = mycv.binarize(pilImage)
    
    
    
    
    #===========================================================================
    # if(himage.label != 1):
    # 
    #    pilImage = picHandler.getRect(pilImage)
    # #print 'newsize = %d %d' % (pilImage.size[0], pilImage.size[1])
    # 
    # 
    # 
    #    pilImage= picHandler.thinning(pilImage)
    # #print newSize
    #    pilImage = pilImage.resize(oriSize, Image.ANTIALIAS)
    #===========================================================================
    himage.image = pilImage
    return himage
    

def writeFile(filename, imageList):
    with open(filename, 'w') as f:
        if(len(imageList) > 0):
            f.write(str(len(imageList)))
            f.write('\r\n')
            
            pilimage = processImage(imageList[0]).image
            
            f.write('%d' % (13))
            f.write('\r\n')
            
            count = 0
            max = len(imageList)
            
            for image in imageList:
                appendImage(f, image)
                if(count % 50 == 0):
                    print "%.3f" % (count / float(max) * 100)
                count += 1

def appendImage(f, himage):
    f.write(str(himage.label))
    f.write('\r\n')
    
    afterHImage = processImage(himage) 
    
    
    
    afterArr = picHandler.divInto13(afterHImage.image)
    
    
    for num in afterArr:
        f.write('%.8f' % (num))
        f.write(' ')
    f.write('\r\n')

def trimArr(arr):
    newArr = []
    for i in range(len(arr)):
        if(arr[i] != 0):
            newArr.append(1)
        else:
            newArr.append(0)
    
    return newArr            

def readData():
    imaList = []
    with open(TRAIN_LABEL, 'rb') as f:
        with open(TRAIN_IMAGE, 'rb') as image:
            type, count = struct.unpack('>ii', f.read(8))
            imaType, imaCount, imaRow, imaCol = struct.unpack('>iiii', image.read(16))
            imageFmt = 'B' * (imaRow * imaCol)
            print count
            print imaType, imaCount, imaRow, imaCol
            
            assert count == imaCount
            
            for i in range(count):
                ima = HImage()
                
                label = struct.unpack('B', f.read(1))
                #print label[0]
                
                imaArr = struct.unpack(imageFmt, image.read((imaRow * imaCol)))
                
                #print imaArr
                ima.label = label[0]
                
                ima.image = Image.new('L', (imaRow, imaCol), 0)
                ima.image.putdata(imaArr)

                imaList.append(ima)
    
    print 'begin to writeBMP'
    for i in range(20):
        ima = imaList[i]
        
        writeBMP(ima, 'test' + str(i) + '_' + str(ima.label))
    
    print 'total = %d' % (len(imaList))
    #writeFile('image.txt', imaList)
        


def main():
    readData()
    
if __name__ == '__main__':
    main()
    
