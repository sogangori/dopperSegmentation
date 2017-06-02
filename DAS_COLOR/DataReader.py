from PIL import Image
import numpy
import numpy as np
from scipy.misc import toimage
import glob
from sklearn.preprocessing import StandardScaler

class DataReader():
    
    folder ="C:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData"
    pathTrain = folder +"/das/*.dat"
    #pathTrain = folder +"/das_threshold/*.dat"
    
    channel = 301
    inC = 300
    w = 256
    h = 256
    startY =24
    dstH = 156
            
    def __init__(self):
        print ("DataReader.py __init__") 
    
    def Normalize(self, src):
        src_shape = np.shape(src)
        if np.ndim(src) > 2:
            print ('ndim',np.ndim(src))
            src2d = np.reshape(src,(-1,np.shape(src)[-1]))
        else:
            src2d = src
        src_normal = StandardScaler().fit_transform(src2d)
        src_back = np.reshape(src_normal,src_shape)
        return src_back
        
    def CutHeight(self, src):
        startY= self.startY
        dstH = self.dstH 
        dst = src[:,startY:startY+dstH ,:]
        return dst

    def GetData(self, count):
        
        w = self.w
        h = self.h
        c = self.inC        
        channel = self.channel
        path = self.pathTrain
       
        list = glob.glob(path)                
        count = numpy.minimum(len(list), count)
        print ("count", count)
        
        setIn = numpy.zeros(shape=(count,h,w,c), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(count,h,w), dtype=numpy.float32) 
            
        for n in range(0, count):
            raw = np.fromfile(list[n], np.float32)  
            print ('raw', count, n, raw.shape)
            array = numpy.reshape(numpy.asarray(raw),[channel,h,w]);            
            
            for ch in range(0, c):
                setIn[n][:,:,ch]= array[1+ch,:]                
            
            setOut[n][:]= array[0,:]
    
        setIn = self.Normalize(setIn)
        setIn = self.CutHeight(setIn)
        setOut = self.CutHeight(setOut)          
        return [setIn, setOut] 

    def Augment(self, src0, src1, aug):
        
        count = src0.shape[0]
        h = src0.shape[1]
        w = src0.shape[2]
        c = src0.shape[3]
        setIn = numpy.zeros(shape=(count*aug,h,w,c), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(count*aug,h,w), dtype=numpy.float32) 
        print ('count',count)
        for n in range(0, count):            
            print ('augment ',n,'/',count)
            setIn[n,:] = setIn_one = src0[n,:]
            setOut[n,:] = setOut_one = src1[n,:]                
            if aug > 1:
                n1 = n+count
                setIn[n1,:]= np.fliplr(setIn_one)
                setOut[n1,:]= np.fliplr(setOut_one) 
            if aug > 2:
                n2 = n+count*2
                setIn[n2,:]= setIn_one[::-1]
                setOut[n2,:]= setOut_one[::-1]
            if aug > 3:
                n3 = n+count*3
                setIn[n3,:]= np.flipud(setIn_one)
                setOut[n3,:]= np.flipud(setOut_one) 
        
        return [setIn, setOut]

    def GetDataAug(self, count, aug):        
        setIn, setOut = self.GetData(count)           
        return self.Augment(setIn,setOut, aug) 
                  
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
  