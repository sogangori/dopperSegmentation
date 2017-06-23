from PIL import Image
import numpy
import numpy as np
from scipy.misc import toimage
import glob
from sklearn.preprocessing import StandardScaler

class DataReader():
    
    folder ="C:/Users/pc/Documents/Visual Studio 2013/Projects/DopplerTrainPreProcess/IQApp_cuda/bin/x64/Debug/trainData"
    #pathTrain = folder +"/das9/*.dat"
    pathTrain = folder +"/das_h416/*.dat"
    #pathTrain = folder +"/das/*.dat"
    #pathTrain = folder +"/das_threshold/*.dat"
    
    channel = 171
    inC = channel - 1
    w = 256
    h = 416
    startY = 0
    dstH = h
    ext = ".png"
            
    def __init__(self):
        print ("DataReader.py __init__") 
    
    def NormalizeAll(self, src):
        src_shape = np.shape(src)
        
        print ('ndim',np.ndim(src))
        src1d = np.reshape(src,[-1])
        
        mean = np.mean(src1d)
        std = np.std(src1d)
        print ('before  mean/std', mean, std)
        src = (src1d)/(std+0.001)
        mean = np.mean(src)
        std = np.std(src)
        print ('after mean/std', mean, std)
        src_back = np.reshape(src,src_shape)
        return src_back

    def Normalize(self, src):
        src_shape = np.shape(src)
        if np.ndim(src) > 2:
            print ('ndim',np.ndim(src))
            src2d = np.reshape(src,(-1,np.shape(src)[-1]))
        else:
            src2d = src
        src_normal = StandardScaler().fit_transform(src2d)

        mean = np.mean(src2d)
        std = np.std(src2d)
        print('normalize src shape', src2d.shape)
        print ('before  mean/std', mean, std)
        mean = np.mean(src_normal)
        std = np.std(src_normal)
        print ('after mean/std', mean, std)
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
            print ('Read file as raw', count, n, raw.shape)
            array = numpy.reshape(numpy.asarray(raw),[channel,h,w]);            
            
            for ch in range(0, c):
                setIn[n][:,:,ch]= array[1+ch,:]                
            
            setOut[n][:]= array[0,:]
                     
        return [setIn, setOut] 

    def Augment(self, src0, src1, aug):
        if aug ==1: return [src0, src1 ]
        if aug > 4: aug=4
        n = src0.shape[0]
        h = src0.shape[1]
        w = src0.shape[2]
        c = src0.shape[3]
        setIn = numpy.zeros(shape=(n*aug,h,w,c), dtype=numpy.float32)
        setOut = numpy.zeros(shape=(n*aug,h,w), dtype=numpy.float32) 
        setIn[:n,:] = src0
        setOut[:n,:] = src1
        for i in range(0, n):            
            print ('augment ',i,'/',n)
            setIn_one = src0[i,:]
            setOut_one = src1[i,:]                
            if aug > 1:
                n1 = i+n
                setIn[n1,:]= np.fliplr(setIn_one)
                setOut[n1,:]= np.fliplr(setOut_one) 
            if aug > 2:
                n2 = i+n*2              
                setIn[n2,:]= np.flipud(setIn_one)
                setOut[n2,:]= np.flipud(setOut_one)
            if aug > 3:
                n3 = i+n*2              
                setIn[n3,:]= np.flipud(setIn[n1,:])
                setOut[n3,:]= np.flipud(setOut[n1,:]) 
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
    
    def Append_ensemble(self,data_in,data_out, ensemble):   
        data_in_a = data_in[:,:,:,0:ensemble]
        data_out_a = data_out
        for i in range(data_in.shape[3] - ensemble):
            data_in_b = data_in[:,:,:,1+i:1+i+ensemble]
            data_in_a = np.append(data_in_a, data_in_b, axis=0)
            data_out_a = np.append(data_out_a, data_out, axis=0)
        return data_in_a,data_out_a

    def GetData3(self, count, aug, ensemble):        
        setIn, setOut = self.GetData(count)        
        #setIn = self.NormalizeAll(setIn)
        #setIn = self.CutHeight(setIn)
        #setOut = self.CutHeight(setOut) 
        count = setIn.shape[3]
        
        offset0 = count - ensemble * 2
        offset1 = count - ensemble * 1

        in_train = setIn[:,:,:,0:offset0]
        in_val = setIn[:,:,:,offset0:offset1]
        in_test = setIn[:,:,:,offset1:]
                
        out_train = out_val = out_test = setOut        
        in_train,out_train = self.Augment(in_train,out_train, aug)

        #half_offset = (int)(offset0/2)
        #in_train_0 = in_train[:,:,:,0:half_offset]
        #in_train_1 = in_train[:,:,:,half_offset:]
        #in_train = np.append(in_train_0,in_train_1,axis=0)
        #out_train = np.append(out_train,out_train,axis=0)
        return in_train,out_train,in_val,out_val, in_test, out_test

    def GetDataS(self, count, aug, ensemble):        
        setIn, setOut = self.GetData(count)        
        count = setIn.shape[3]
        
        offset0 = count - ensemble * 3
        offset1 = count - ensemble * 1

        in_train = setIn[:,:,:,0:offset0]
        in_val = setIn[:,:,:,offset0:offset1]
        in_test = setIn[:,:,:,offset1:]
                
        out_train = out_val = out_test = setOut
        half_offset = (int)(offset0/2)
        in_train_0 = in_train[:,:,:,0:half_offset]
        in_train_1 = in_train[:,:,:,half_offset:]
        in_train = np.append(in_train_0,in_train_1,axis=0)
        out_train = np.append(out_train,out_train,axis=0)

        in_val,out_val = self.Append_ensemble(in_val,out_val,ensemble)
        in_test,out_test = self.Append_ensemble(in_test,out_test,ensemble)

        in_train,out_train = self.Augment(in_train,out_train, aug)
        return in_train,out_train,in_val,out_val, in_test, out_test

    def GetNextBatch(self):

        return 0
    
    def SaveAsImage(self, src, filePath, count = 1):
        ext = ".png"        
        print ("SaveAsImage","count:", count, src.size, src.shape, filePath, ext)
        src = numpy.reshape(src, [count,self.dstH,self.w])
        for i in range(0, count):
            
            img = toimage(src[i,:])
            fileName =filePath+ str(i) +ext
            img.save( fileName )    
            
    def SaveAsImageByChannel(self, src, filePath, count = 1):
        
        print ("SaveAsImageByChannel","count:", count, src.shape, filePath,ext   )
        
        for i in range(0, count):          
            for c in range(0, src.shape[3]):
                inChannel = numpy.abs(src[i,:,:,c])
                
                #img = toimage(inChannel/numpy.max(inChannel)*255)                                
                img = toimage(inChannel*255)
                fileName =filePath+ str(i)+"_"+ str(c)+"_tri"+self.ext  
                img.save( fileName ) 
    
    def SaveImage(self, src, filePath):                
        print ("SaveTensorImage",  src.shape, filePath)
          
        for i in range(0, src.shape[0]):  
            img = toimage( src[i,:])
            fileName =filePath+"_t_"+ str(i)+self.ext 
            img.save( fileName )
             
    def SaveImageNormalize(self, src, filePath):
        print ("SaveTensorImage",  src.shape, filePath)
          
        for i in range(0, src.shape[0]):  
            data = src[i,:]
            src_shape = data.shape
            data_2d = np.reshape(data, (-1, data.shape[np.ndim(data)-1]) )            
            data_normal = data_2d/np.max(data_2d,0)
            data_normal_2d = np.reshape(data_2d, src_shape)
            img = toimage(data_normal_2d)
            fileName =filePath+"_t_"+ str(i)+self.ext  
            img.save( fileName ) 
                     
    def SNR(self, label, predict):
        print ('SNR min,max',numpy.min(label),numpy.max(label),'predict minMax Mean',numpy.min(predict), numpy.max(predict), numpy.mean(predict))
        noise = label - predict
        noiseSum = numpy.sum(noise*noise)
        signalSum =numpy.sum(numpy.abs(label))
        print ('noiseSum',noiseSum,'signalSum',signalSum)
        snr = 10 * numpy.log10(signalSum/noiseSum)
        return snr                      
  