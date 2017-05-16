

from PIL import Image
import numpy
import numpy as np
from scipy import ndimage
from scipy.misc import toimage
import glob
from sklearn.metrics import confusion_matrix
class DataReader():
    
    pathTrain = "C:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData/in3CH_float360/*.dat"    
    channel = 5
    inC = 3
    w = 256
    h = 256
                       
    def __init__(self):
        print ("TrainData __init__") 
          
    def ConvertToTrimap(self, src):
        h = src.shape[0]
        w = src.shape[1]
        kernel = (7,7)
        allOne = np.ones([h,w])
        dilat = ndimage.grey_dilation(src, size=kernel)        
        foreground = ndimage.grey_erosion(src, size=kernel)        
        unknown = dilat-foreground
        tripmap =  unknown * 0.5 + foreground
        return tripmap

    
    def GetTrainDataToTensorflowTest(self, isTrain = True,isPatch = False):
        w = self.w
        h = self.h        
        channel = self.channel      
        path = self.pathTrain              
       
        list = glob.glob(path)
        count = 1
        
        setIn = numpy.zeros(shape=(count,h,w,self.inC), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(count,h,w), dtype=numpy.float32) 
            
        for n in range(0, count):
            raw = np.fromfile(list[n+310], np.float32)  
            print ('raw', raw.shape          )
            array = numpy.reshape(numpy.asarray(raw),[channel,h,w]);            
            
            for ch in range(0, self.inC):                    
                setIn[n][:,:,ch]= array[ch,:,:]   
            setOut[n][:]= array[3,:,:]
            #setOut[n][:]= self.ConvertToTrimap(setOut[n][:])
        return [setIn, setOut]    
  
    def GetTrainDataToTensorflow(self, batchCount, isTrain = True, offset = 0):
        
        path = self.pathTrain     
        channel = self.channel
        w = self.w
        h = self.h        
        
        list = glob.glob(path)
        #numpy.random.shuffle(list)
        count = len(list)
        if count > batchCount: 
            count = batchCount;                
        
        setIn = numpy.zeros(shape=(count,h,w,self.inC), dtype=numpy.float32) 
        setOut = numpy.zeros(shape=(count,h,w), dtype=numpy.float32) 
               
        for n in range(0, count):
            raw = np.fromfile(list[n + offset], np.float32)            
            array = numpy.reshape(numpy.asarray(raw),[channel,h,w]);
            
            for ch in range(0, self.inC):                    
                setIn[n][:,:,ch]= array[ch,:,:]
            
            label = array[3,:,:]            
            setOut[n][:,:]= label
                    
        return [setIn, setOut]    
    
    def GetTrainDataToTensorflowRotateLR(self, batchCount, aug = 1, isTrain = True):
        
        path = self.pathTrain        
        channel = self.channel
        if not isTrain:
            path = self.pathTest
            self.h = 256                    
        w = self.w
        h = self.h        
        
        list = glob.glob(path)
        #numpy.random.shuffle(list)
        count = len(list)
        if count > batchCount: 
            count = batchCount;                
        
        setIn = numpy.zeros(shape=(count*aug,h,w,self.inC), dtype=numpy.float32) 
        setOut = numpy.zeros(shape=(count*aug,h,w), dtype=numpy.int32) 
        
        for n in range(0, count):
            raw = np.fromfile(list[n], np.float32)            
            array = numpy.reshape(numpy.asarray(raw),[channel,h,w]);
            label = array[3,:,:]
            
            n0 = n*aug
            n1 = n*aug+1
            n2 = n*aug+2
            for ch in range(0, self.inC):
                               
                setIn[n0][:,:,ch]= array[ch,:,:]
                setOut[n0][:,:]= label
                
                if aug>=2:
           
                    for y in range(h):
                        for x in range(w):                        
                            index_y = y                        
                            index_x = w-1-x                       
                            setIn[n1][y,x,ch]= array[ch,index_y,index_x]
                            label_v = label[index_y,index_x]
                            setOut[n1][y,x]= label_v 
                            if aug>=3:
                                index_x2 = x                        
                                index_y2 = h-1-y                        
                                setIn[n2][y,x,ch]= array[ch,index_y2,index_x2]
                                label_v2 = label[index_y2,index_x2]
                                setOut[n2][y,x]= label_v2 
               
                    setIn[n1][:,:,ch] = setIn[n1][:,:,ch]
            if aug>=3: setIn[n2][:,:,ch] = setIn[n2][:,:,ch]
        return [setIn, setOut]   
  
    
    def SaveAsImageSoftmax(self, src, filePath, count = 1):
        ext = ".png"        
        count = src.shape[0]/self.h/self.w
        src = numpy.reshape(src, [count,self.h,self.w])
             
        for i in range(0, count):
            img = toimage(src[i,:,:,0])
            fileName =filePath+ str(i) +ext
            img.save( fileName )
            
    def SaveAsImage(self, src, filePath, count = 1):
        ext = ".png"     
        print ('SaveAsImage', src.shape)
        count = src.shape[0]
        #src = numpy.reshape(src, [count,-1,self.w])
         
        for i in range(0, count):
            img = toimage(src[i,:])
            fileName =filePath+ str(i) +ext
            img.save( fileName )    
            
    def SaveAsImageByChannel(self, src, filePath, count = 1):
        ext = ".png"        
        
        for i in range(0, count):          
            for c in range(0, src.shape[3]):
                inChannel = numpy.abs(src[i,:,:,c])
                
                #img = toimage(inChannel/numpy.max(inChannel)*255)                                
                img = toimage(inChannel*255)
                fileName =filePath+ str(i)+"_"+ str(c)+ext 
                img.save( fileName ) 
    
    def SaveTensorImage(self, src, filePath):
        ext = ".png"        
          
        for i in range(0, src.shape[0]):  
            if i<1:        
                for c in range(0, src.shape[3]):
                    inChannel = numpy.abs(src[i,:,:,c])
                    
                    img = toimage(inChannel/numpy.max(inChannel)*255)                                
                    #img = toimage(inChannel*255)
                    fileName =filePath+"_"+ str(c)+ext 
                    img.save( fileName ) 
                            
    def accuracy2(self, out, predict):
        wh =out.shape[1]*out.shape[2]
        #print out.shape, predict.shape
        
        count = out.shape[0]        
        predict = numpy.reshape(predict, [count,wh])
        out = numpy.reshape(out, [count,wh])
        predictMax = numpy.max(predict)
        #print 'accuracy2 min,max',numpy.min(out),numpy.max(out),'predict minMax',numpy.min(predict),predictMax 
        #print 'confusion_matrix'
        #print( confusion_matrix(out, predict))
        
        recall = 0.0    
        precision = 0.0
        precisionSum = 1.0        
        offset = predictMax / 2
        
        for i in range(0, count):
            for j in range(0, wh):
                if out[i,j] > offset and  predict[i,j]>offset:
                    recall += 1
                if predict[i,j] > offset:
                    precisionSum +=1
                    if out[i,j] >offset:
                        precision += 1
                
        return precision/precisionSum, recall/(numpy.sum(out)+1)
    
    def accuracy(self, out, predict):
        print ('accuracy min,max',numpy.min(out),numpy.max(out),'predict minMax',numpy.min(predict),numpy.max(predict))
        src =  numpy.round(out / numpy.max(out))
        predict = numpy.round(predict / numpy.max(predict))
                
        recall = 0.0    
        precision = 0.0
        precisionSum = 0.0
        size = 1
        offset = 0.1
        for dim in numpy.shape(out): size *= dim
        
        src1 = numpy.reshape(src, size);
        predict1 = numpy.reshape(predict, size );        
        for i in range(0, size):
            if src1[i] > offset and  predict1[i]>offset:
                recall += 1
            if predict1[i] > offset:
                precisionSum +=1
                if src1[i] >offset:
                    precision += 1
 
        return precision/precisionSum, recall/(numpy.sum(src1)+0.1)      
     