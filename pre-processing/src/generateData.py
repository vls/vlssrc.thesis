# -*- coding: utf-8 -*-
from __future__ import with_statement
from PIL import Image
import sys
import struct
import mycv


TRAIN_LABEL = 'data/The MNIST database of handwritten digits/train-labels.idx1-ubyte'
TRAIN_IMAGE = 'data/The MNIST database of handwritten digits/train-images.idx3-ubyte'

class HImage():
    def __init__(self):
        self.label = -1
        self.row = 0
        self.col = 0
        self.arr = []
        
def writeBMP(himage, filename):
    imn = Image.new('L', (himage.row, himage.col), 0)
    imn.putdata(himage.arr)
    after = mycv.binarize(imn)
    after.save(filename + '_.bmp', 'BMP')
    #print list(after.getdata())
    imn.save(filename + '.bmp', 'BMP')

def binarizeArr(arr, row, col):
    imn = Image.new('L', (row, col), 0)
    imn.putdata(arr)
    after = mycv.binarize(imn)
    afterArr = list(after.getdata())
    return afterArr
    

def writeFile(filename, imageList):
    with open(filename, 'w') as f:
        if(len(imageList) > 0):
            f.write(str(len(imageList)))
            f.write('\r\n')
            f.write('%d %d' % (imageList[0].row, imageList[0].col))
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
    afterArr = binarizeArr(himage.arr, himage.row, himage.col)
    afterArr = trimArr(afterArr)
    for num in afterArr:
        f.write(str(num))
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
                ima.arr = imaArr
                ima.row = imaRow
                ima.col = imaCol
                imaList.append(ima)
    
    
    for i in range(10):
        ima = imaList[i]
        writeBMP(ima, 'test' + str(i) + '_' + str(ima.label))
    
    print 'total = %d' % (len(imaList))
    writeFile('image.txt', imaList)
        


def main():
    readData()
    
if __name__ == '__main__':
    main()
    
