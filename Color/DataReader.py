from PIL import Image
import numpy
import numpy as np
import random
from scipy.misc import toimage
from scipy import ndimage
import glob
from skimage import measure
from skimage import filters

class DataReader():
        
    folder = "C:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData"
    pathTrain =  folder + "/in3CH_float360/*.dat"    
    pathTrain0 = folder + "/in3CH_float/*.dat"
    pathTrain1 = folder + "/in3CH_float_blur/*.dat"
    pathTest =   folder + "/in3CH_float_blur_test/*.dat"
        
    dataC = 5
    c = 3
    w = 256
    h = 256
            
    def __init__(self):
        print ("DataReader.py __init__") 

    def ConvertToTrimap(self, src):
        h = src.shape[0]
        w = src.shape[1]
        kernel = (3,3)
        allOne = np.ones([h,w])
        dilat = ndimage.grey_dilation(src, size=kernel)        
        foreground = ndimage.grey_erosion(src, size=kernel)        
        unknown = dilat-foreground
        tripmap =  unknown * 0.5 + foreground
        return tripmap
    
    def ConvertToUnknown(self, src):
        h = src.shape[0]
        w = src.shape[1]
        kernel = (5,5)        
        dilat = ndimage.grey_dilation(src, size=kernel)        
        foreground = ndimage.grey_erosion(src, size=kernel)        
        unknown = dilat-foreground        
        return unknown
    
    
    def FindMask(self,src):
        sum_cols = np.sum(src, axis=1)
        sum_rows = np.sum(src, axis=0)
        w = len(sum_cols)
        h = len(sum_rows)
        feature0=[-1,h,-1,w]
        for i in range(len(sum_cols)):
            if feature0[0]==-1 and sum_cols[i] > 0:
                feature0[0] = i
            elif feature0[0]>-1 and sum_cols[i] == 0:
                feature0[1] = i
                break
    
        for i in range(len(sum_rows )):
            if feature0[2]==-1 and sum_rows [i] > 0:
                feature0[2] = i
            elif feature0[2]>-1 and sum_rows [i] == 0:
                feature0[3] = i
                break
        return feature0

    def ConvertToROI(self,src):
        blobs = src > src.mean()
        all_labels = measure.label(blobs)
        blobs_labels = measure.label(blobs, background=0)
        for i in range(np.max(all_labels)):
            data1 = np.array(all_labels==i+1,dtype = int)
            idx = self.FindMask(data1)
            all_labels[idx[0]:idx[1],idx[2]:idx[3]] = 1
        return all_labels
        

    def ConvertToDilateROI(self,src):
        kernel = (5,5)        
        dilat = ndimage.grey_dilation(src, size=kernel)
        dilat = ndimage.grey_dilation(dilat, size=kernel)                
        return dilat

    def GetData(self, count, isTrain = True):
        
        w = self.w
        h = self.h        
        c = self.c
        path = self.pathTest
        if isTrain:
            path = self.pathTrain
      
        list = glob.glob(path)
      
        random.shuffle(list)    
        print ("count", count,list[0])
        
        setIn = numpy.zeros(shape=(count,h,w,c), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(count,h,w), dtype=numpy.float32) 
            
        for n in range(0, count):
            raw = np.fromfile(list[n], np.float32)  
            
            array = numpy.reshape(numpy.asarray(raw),[self.dataC,h,w]);            
            #print ('raw', raw.shape, array.shape)
            for ch in range(c):                                                                                                 
                setIn[n,:,:,ch]= array[ch,:,:]

            setOut[n][:]= array[c,:,:]            
        
        return [setIn, setOut]
    
    def GetDataAug(self, count, aug = 2, isTrain = True):
        
        w = self.w
        h = self.h        
        c = self.c
        path = self.pathTrain
        if not isTrain:
            path = self.pathTest
      
        list = glob.glob(path)    
        if isTrain:
            list0 = glob.glob(self.pathTrain0)
            list1 = glob.glob(self.pathTrain1)            
            list.extend(list0)
            list.extend(list1)
            print ("[INFO] append count ", count,list[0])
            random.shuffle(list)  
        if len(list) < count : count = len(list)
        print ("aug, limit count " ,aug, count,list[0])
        setIn = numpy.zeros(shape=(count*aug ,h,w,c), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(count*aug ,h,w), dtype=numpy.float32) 
        setHelper = numpy.zeros(shape=(count*aug ,h,w), dtype=numpy.float32) 
        for n in range(count):
            raw = np.fromfile(list[n], np.float32)  
            n0 = n*aug
            array = numpy.reshape(numpy.asarray(raw),[self.dataC,h,w]);          
            #print ('raw', raw.shape, array.shape)
            for ch in range(c):                                                                                     
                setIn[n0,:,:,ch]= array[ch,:,:]            
            setOut[n0,:]= array[c,:]

        for n in range(count):
            n0 = n*aug
            n1 = n*aug+1
            n2 = n*aug+2
            n3 = n*aug+3
            if aug > 1:
                setIn[n1,:]= np.fliplr(setIn[n0,:])
                setOut[n1,:]= np.fliplr(setOut[n0,:])            
            if aug > 2:
                setIn[n2,:]= np.flipud(setIn[n0,:])
                setOut[n2,:]= np.flipud(setOut[n0,:])   
            if aug > 3:
                setIn[n3,:]= np.flipud(setIn[n1,:])
                setOut[n3,:]= np.flipud(setOut[n1,:])
            if aug > 4:
                setIn[n*aug+4,:]= np.rot90(setIn[n0,:])
                setOut[n*aug+4,:]= np.rot90(setOut[n0,:])
            if aug > 5:
                setIn[n*aug+5,:]= np.rot90(setIn[n1,:])
                setOut[n*aug+5,:]= np.rot90(setOut[n1,:])      
            if aug > 6:
                setIn[n*aug+6,:]= np.rot90(setIn[n2,:])
                setOut[n*aug+6,:]= np.rot90(setOut[n2,:])
            if aug > 7:
                setIn[n*aug+7,:]= np.rot90(setIn[n3,:])
                setOut[n*aug+7,:]= np.rot90(setOut[n3,:])

        for n in range(count*aug):
            setHelper[n] = self.ConvertToUnknown(setOut[n,:])            

        return [setIn, setOut, setHelper] 
    
    def GetDataROI(self, count, aug = 2, isTrain = True):
        
        w = self.w
        h = self.h        
        c = self.c
        path = self.pathTrain
        if not isTrain:
            path = self.pathTest
      
        list = glob.glob(path)    
        if isTrain:            
            list.extend(glob.glob(self.pathTrain0))
            #list.extend(glob.glob(self.pathTrain1))
            print ("[INFO] append count ", count,list[0])
            random.shuffle(list)  
        if len(list) < count : count = len(list)
        print ("aug, limit count " ,aug, count,list[0])
        setIn = numpy.zeros(shape=(count*aug ,h,w,c), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(count*aug ,h,w), dtype=numpy.float32) 
        setHelper = numpy.zeros(shape=(count*aug ,h,w), dtype=numpy.float32) 
        for n in range(count):
            raw = np.fromfile(list[n], np.float32)  
            n0 = n*aug
            array = numpy.reshape(numpy.asarray(raw),[self.dataC,h,w]);          
            #print ('raw', raw.shape, array.shape)
            for ch in range(c):                                                                                     
                setIn[n0,:,:,ch]= array[ch,:,:]            
            setOut[n0,:]= array[c,:]

        for n in range(count):
            n0 = n*aug
            n1 = n*aug+1
            n2 = n*aug+2
            n3 = n*aug+3
            if aug > 1:
                setIn[n1,:]= np.fliplr(setIn[n0,:])
                setOut[n1,:]= np.fliplr(setOut[n0,:])            
            if aug > 2:
                setIn[n2,:]= np.flipud(setIn[n0,:])
                setOut[n2,:]= np.flipud(setOut[n0,:])   
            if aug > 3:
                setIn[n3,:]= np.flipud(setIn[n1,:])
                setOut[n3,:]= np.flipud(setOut[n1,:])
            if aug > 4:
                setIn[n*aug+4,:]= np.rot90(setIn[n0,:])
                setOut[n*aug+4,:]= np.rot90(setOut[n0,:])
            if aug > 5:
                setIn[n*aug+5,:]= np.rot90(setIn[n1,:])
                setOut[n*aug+5,:]= np.rot90(setOut[n1,:])      
            if aug > 6:
                setIn[n*aug+6,:]= np.rot90(setIn[n2,:])
                setOut[n*aug+6,:]= np.rot90(setOut[n2,:])
            if aug > 7:
                setIn[n*aug+7,:]= np.rot90(setIn[n3,:])
                setOut[n*aug+7,:]= np.rot90(setOut[n3,:])

        for n in range(count*aug):
            setHelper[n] = self.ConvertToROI(setOut[n,:])           

        return [setIn, setOut, setHelper] 
    def GetDataCheck(self, count, aug = 2, isTrain = True):
        
        w = self.w
        h = self.h        
        c = self.c
        path = self.pathTrain
        if not isTrain:
            path = self.pathTest
      
        list = glob.glob(path)    
        random.shuffle(list)    
        print ("aug, limit count " ,aug, count,list[0])
        setIn = numpy.zeros(shape=(count*aug ,h,w,c), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(count*aug ,h,w), dtype=numpy.float32) 
        setCompare = numpy.zeros(shape=(count*aug ,h,w), dtype=numpy.float32) 
        for n in range(0, count):
            raw = np.fromfile(list[n], np.float32)  
            n0 = n*aug
            n1 = n*aug+1
            array = numpy.reshape(numpy.asarray(raw),[self.dataC,h,w]);          
            #print ('raw', raw.shape, array.shape)
            for ch in range(c):                                                                                     
                setIn[n0,:,:,ch]= array[ch,:,:]
            
            setOut[n0,:]= array[c,:]
            setCompare[n0,:]= array[c+1,:]

            if aug > 1:
                setIn[n1,:]= np.fliplr(setIn[n0,:])
                setOut[n1,:]= np.fliplr(setOut[n0,:])
                setCompare[n1,:]= np.fliplr(setCompare[n0,:])                            
            if aug > 2:
                setIn[n*aug+2,:]= np.flipud(setIn[n0,:])
                setOut[n*aug+2,:]= np.flipud(setOut[n0,:])
                setCompare[n*aug+2,:]= np.flipud(setCompare[n0,:])   
            if aug > 3:
                setIn[n*aug+3,:]= np.flipud(setIn[n1,:])
                setOut[n*aug+3,:]= np.flipud(setOut[n1,:])
                setCompare[n*aug+3,:]= np.flipud(setCompare[n1,:])             
        return [setIn, setOut, setCompare] 
    
 
    def SaveAsImage(self, src, filePath, count, maxCount=1):
        ext = ".png"        
        print ("SaveAsImage","maxCount:", maxCount, src.shape, filePath, ext)
        
        for i in range(0, np.minimum( count,maxCount)):            
            img = toimage(src[i,:])
            fileName =filePath+ str(i) +ext
            img.save( fileName )    
            
    def SaveAsImageByChannel(self, src, filePath, count = 1):
        ext = ".png"        
        print ("SaveAsImageByChannel","count:", count, src.shape, filePath,ext   )
        
        for i in range(0, count):          
            for c in range(0, src.shape[3]):
                inChannel = numpy.abs(src[i,:,:,c])
                
                #img = toimage(inChannel/numpy.max(inChannel)*255)                                
                img = toimage(inChannel*255)
                fileName =filePath+ str(i)+"_"+ str(c)+ext 
                img.save( fileName ) 

    def SaveFeatureMap(self, srcList, filePath, count, maxCount=1):
        ext = ".png"        
        
        print ("SaveFeatureMap","count","maxCount:",count, maxCount, len(srcList), filePath, ext)
        for n in range(len(srcList)):
            bach_data = np.array( srcList[n])
            for i in range(maxCount):
                one_data = bach_data[i,:]
                shape = one_data.shape
                #print("featureMap ",n,bach_data.shape,shape)
                fileName =filePath+ str(i)+"_"+ str(n)+ ext 
                data_reshape = np.reshape(one_data, [-1,one_data.shape[2]] )
                #print ("before max",numpy.max(data_reshape , axis = 0))
                one_data = data_reshape/numpy.max(data_reshape, axis = 0)*255
                #print ("after  max",numpy.max(one_data, axis = 0))
                one_data = np.reshape(one_data, shape )
                img = toimage(one_data)                                
                img.save( fileName ) 
                
        