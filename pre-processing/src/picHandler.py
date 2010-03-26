# -*- coding: utf-8 -*-

from PIL import Image

def getPos(x, y, col):
    return x* col + y

def getRect(pilimage):
    matrix = list(pilimage.getdata())
    col = pilimage.size[1]
    row = pilimage.size[0]
    
    left = col -1
    right = 0
    top = row - 1
    bottom = 0
    
    for i in range(row):
        for j in range(col):
            if(matrix[getPos(i,j,col)] > 0):
                left = min(j, left)
                right = max(j, right)
                top = min(i, top)
                bottom = max(i, bottom)
    
    left = left - 1 < 0 and left or left -1
    right = right + 1 >= col and right or right + 2
    top = top -1 < 0 and top or top -1
    bottom = bottom + 1 >= row and bottom or bottom +2
    print (left, top, right, bottom)
    print (row, col)
    return pilimage.crop((left, top, right, bottom))




def thinning(pilimage):
    matrix = list(pilimage.getdata())
    row = pilimage.size[1]
    col = pilimage.size[0]
    #print matrix
    #find the edge
    
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    
    
    def __getS(x, y, mat):
        roundList = getRoundList(x, y)
        if(len(roundList) != 8):
            return None
        
        
    
    def isValid(x, y):
        return x>= 0 and x < row and y>=0 and y<col
    
    def getRoundList(x, y):
        roundList = []
        for i in range(8):
            newx = x+dx[i]
            newy = y+dy[i]
            #print "newx = %d, newy =%d, valid = %s" % (newx, newy, isValid(newx, newy))
            if(isValid(newx, newy)):
                roundList.append((newx, newy))
        #=======================================================================
        # if(len(roundList) != 8):
        #    for i in range(8):
        #        newx = x+dx[i]
        #        newy = y+dy[i]
        #        if(not isValid(newx, newy)):
        #            print "not valid newx = %d, newy = %d / x = %d, y = %d / col = %d, row = %d" % (newx, newy, x, y, col, row)
        #=======================================================================
                        
        return roundList
    
      
    
    def isEdge(x, y):
        if(matrix[getPos(x, y, col)] > 0):
            roundList = getRoundList(x, y)      
            for newx,newy in roundList:
                pos = getPos(newx, newy, col)
                if(matrix[pos] == 0):
                    return True
        return False
        
    
    def isContourP(x, y, mat, secIter = False):
   
        roundList = getRoundList(x, y)
        
        p = []
        p.append(-1)
        p.append(mat[getPos(x,y,col)])
        #print "%d, %d" % (x, y)
        for newx, newy in roundList:
            #print 'newx = %d, newy = %d %s' % (newx, newy, mat[getPos(newx, newy, col)] > 0)
            p.append(mat[getPos(newx, newy, col)] > 0 and True or False)
        
        #print p
        
        N = 0
        S = 0
        for i in range(2, 10):
            N += (p[i] and [1] or [0])[0]
            k = (i < 9 and [i+1] or [2])[0]
            if((p[k] and not p[i] )):
                S += 1
        
        
        if(not secIter):
        
            if(((p[2] and p[8] or p[4] or p[6]) and N >=2 and N <= 6 and S == 1)):
                print 'wipe %d, %d' % (x, y)
                mat[getPos(x, y, col)] = 0
                return 1
        #=======================================================================
        # else:
        #    if(((p[4] and p[7] or p[2] or p[8]) and N >=2 and N <= 6 and S == 1)):
        #        print 'wipe %d, %d' % (x, y)
        #        mat[getPos(x, y, col)] = 0
        #        return 1
        #=======================================================================
                                 
        
        return 0
    
    while True:
        wipecount = 0
        for i in range(1, row-1):
            for j in range(1, col-1):
                if(matrix[getPos(i,j,col)] > 0):
                    wipecount += isContourP(i, j, matrix)
        if(wipecount == 0):
            print 'one sample done'
            break
        else:
            print 'wipe = %d' %(wipecount)
    
    #print matrix
    icopy = pilimage.copy()
    icopy.putdata(matrix)
    print icopy
    return icopy

patternDict = None

class Pattern:
    def __init__(self):
        self.imageArr = None
        self.totalNum = 0

def __initDict():
    
    pDict = {}
    for i in range(13):
        str = 'pattern/%d.bmp' % (i+1)
        image = Image.open(str)
        p = Pattern()
        p.imageArr = list(image.getdata())
        total = 0
        for pixel in p.imageArr:
            if(pixel > 0):
                total += 1
             
        p.totalNum = total
        pDict[i+1]  = p
    #print pDict
    return pDict

def __getInBlock(pilimage, imageArr):
    
    samArr = list(pilimage.getdata())
    
    if(len(samArr) != len(imageArr)):
        raise Exception('The sizes of two images are different')
    
    count = 0
    length = len(imageArr)
    for i in range(length):
        if(imageArr[i] > 0 and samArr[i] > 0):
            count += 1
        
    return count
    

def divInto13(pilimage):
    
    global patternDict
    if(patternDict == None):
        print 'init dict'
        patternDict = __initDict()
    
    ratioList = []
    
    for i in range(13):
        p = patternDict[i+1]
        imageArr = p.imageArr
        count = __getInBlock(pilimage, imageArr)
        
        ratio = float(count) / p.totalNum
        ratioList.append(ratio)
    
    return ratioList
        
def main():
    divInto13(None)

if __name__ == '__main__':
    main()