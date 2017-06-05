from PIL import Image
import numpy
import numpy as np
from scipy.misc import toimage
import glob
from sklearn.preprocessing import StandardScaler

class DataReader():
    
    folder ="C:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData"
    pathTrain = folder +"/das9/*.dat"
    #pathTrain = folder +"/das/*.dat"
    #pathTrain = folder +"/das_threshold/*.dat"
    
    channel = 301
    inC = 300
    w = 256
    h = 256
    startY =20
    dstH = 128+32
            
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
        
        n = src0.shape[0]
        h = src0.shape[1]
        w = src0.shape[2]
        c = src0.shape[3]
        setIn = numpy.zeros(shape=(n*aug,h,w,c), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(n*aug,h,w), dtype=numpy.float32) 
        for i in range(0, n):            
            print ('augment ',i,'/',n)
            setIn[i,:] = setIn_one = src0[i,:]
            setOut[i,:] = setOut_one = src1[i,:]                
            if aug > 1:
                n1 = i+n
                setIn[n1,:]= np.fliplr(setIn_one)
                setOut[n1,:]= np.fliplr(setOut_one) 
            if aug > 2:
                n2 = i+n*2
                setIn[n2,:]= setIn_one[::-1]
                setOut[n2,:]= setOut_one[::-1]
            if aug > 3:
                n3 = i+n*3
                setIn[n3,:]= np.flipud(setIn[n2,:])
                setOut[n3,:]= np.flipud(setOut[n2,:])           
            if aug > 4:
                n4 = i+n*4
                setIn[n4,:]= np.fliplr(setIn[n2,:])
                setOut[n4,:]= np.fliplr(setOut[n2,:]) 
            if aug > 5:
                n5 = i+n*5
                setIn[n5,:]= np.flipud(setIn_one)
                setOut[n5,:]= np.flipud(setOut_one) 
        return [setIn, setOut]

    def GetDataAug(self, n, aug):        
        setIn, setOut = self.GetData(n)           
        return self.Augment(setIn,setOut, aug) 

    def GetDataTrainTest(self, count, aug, ensemble):        
        setIn, setOut = self.GetData(count)        
        count = setIn.shape[3]
        
        count_train = count - ensemble
        train0 = setIn[:,:,:,0:count_train]
        train1 = setOut
        test0 = setIn[:,:,:,count_train:]
        test1 = setOut

        train0,train1 = self.Augment(train0,train1, aug)
        print ('train_in',train0.shape)
        print ('train_out',train1.shape)
        print ('test_in',test0.shape)
        print ('test_out',test1.shape)        
        return train0,train1,test0,test1
                 
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
        ext = "_tri.png"        
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
            data = src[i,:]
            src_shape = data.shape
            data_2d = np.reshape(data, (-1, data.shape[np.ndim(data)-1]) )            
            data_normal = data_2d/np.max(data_2d,0)
            data_normal_2d = np.reshape(data_2d, src_shape)
            img = toimage(data_normal_2d)
            fileName =filePath+"_t_"+ str(i)+ext 
            img.save( fileName ) 
    def SNR(self, label, predict):
        print ('SNR min,max',numpy.min(label),numpy.max(label),'predict minMax Mean',numpy.min(predict), numpy.max(predict), numpy.mean(predict))
        noise = label - predict
        noiseSum = numpy.sum(noise*noise)
        signalSum =numpy.sum(numpy.abs(label))
        print ('noiseSum',noiseSum,'signalSum',signalSum)
        snr = 10 * numpy.log10(signalSum/noiseSum)
        return snr                      
  