from PIL import Image
import numpy
import numpy as np
from scipy.misc import toimage
import glob

class DataReader():
    
    folder = "C:/Users/pc/Documents/Visual Studio 2013/Projects/DASTrain/x64/Release/Cut/height 974"
    pathTrain = folder +"/train/*.dat"
    pathTest = folder + "/test/*.dat"
    
    channel = 256
    inC = 255
    w = 128
    h = 1024
            
    def __init__(self):
        print ("DataReader.py __init__") 
      
    def GetTrainDataToTensorflow(self, count, isTrain = True, trainH = 128):
        
        w = self.w
        h = self.h        
        channel = self.channel
        path = self.pathTest
        if isTrain:
            path = self.pathTrain
      
        list = glob.glob(path)        
        print ("count", count,list[0])
        
        setIn = numpy.zeros(shape=(count,trainH,w,self.inC), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(count,trainH,w), dtype=numpy.float32) 
            
        for n in range(0, count):
            raw = np.fromfile(list[n], np.float32)  
            print ('raw', raw.shape)
            array = numpy.reshape(numpy.asarray(raw),[channel,h,w]);            
            
            for ch in range(0, self.inC):                          
                
                for y in range(trainH):
                    for x in range(w):                        
                        setIn[n][y,x,ch]= array[1+ch,y,x]
                        setOut[n][y,x]= array[0,y,x]
            
                        
            labelMax=numpy.abs(numpy.max(setOut[n][:]))   
            labelMin=numpy.abs(numpy.min(setOut[n][:]))  
            labelMax = max(labelMax,labelMin)
            setOut[n][:]= setOut[n][:]/labelMax
        return [setIn, setOut]
    
    def GetTrainDataToTensorflowRotateLR(self, batchCount, aug = 1, isTrain = True):
        print ('GetTrainDataToTensorflowRotateLR', isTrain)
        
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
        print ("Original count", count, "batchCount",batchCount, path,h,w )
        if count > batchCount: 
            count = batchCount;                
        
        setIn = numpy.zeros(shape=(count*aug,h,w,self.inC), dtype=numpy.float32) 
        setOut = numpy.zeros(shape=(count*aug,h,w), dtype=numpy.int32) 
        print ("count * aug", count,aug)
        
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
                  
    def SaveAsImage(self, src, filePath, count = 1):
        ext = ".png"        
        print ("SaveAsImage","count:", count, src.shape, filePath, ext)
        
        #src = numpy.abs(src)
        src = numpy.reshape(src, [count,-1,self.w])
        for i in range(0, count):
            
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
    
    def SaveTensorImage(self, src, filePath):
        ext = ".png"        
        print ("SaveTensorImage",  src.shape, filePath,ext)
          
        for i in range(0, src.shape[0]):  
            if i<1:        
                for c in range(0, src.shape[3]):
                    inChannel = numpy.abs(src[i,:,:,c])
                    
                    img = toimage(inChannel/numpy.max(inChannel)*255)                                
                    #img = toimage(inChannel*255)
                    fileName =filePath+"_"+ str(c)+ext 
                    img.save( fileName ) 
    def SNR(self, label, predict):
        print ('SNR min,max',numpy.min(label),numpy.max(label),'predict minMax Mean',numpy.min(predict), numpy.max(predict), numpy.mean(predict))
        noise = label - predict
        noiseSum = numpy.sum(noise*noise)
        signalSum =numpy.sum(numpy.abs(label))
        print ('noiseSum',noiseSum,'signalSum',signalSum)
        snr = 10 * numpy.log10(signalSum/noiseSum)
        return snr                      
  