from DataReader import DataReader
import numpy as np

trainData = DataReader();

def Test_Aug():
    inData,outData = trainData.GetDataAug(9,1);

    print (inData.shape)
    print (outData.shape)

    for n in range(0, inData.shape[0]):
        print ("inData ",n, " max ",np.max(inData[n]), ' min ' , np.min(inData[n]), ' mean ',  np.mean(inData[n]))
        print ("outData ",n," max ", np.max(outData[n]), ' min ' ,np.min(outData[n]), ' mean ', np.mean(outData[n]))
        
    trainData.SaveAsImage(outData, "./ROI/weights/out",outData.shape[0])
    #trainData.SaveAsImageByChannel(inData, "./DAS_COLOR/weights/in",)

def Test_validate():
    in_train,out_train,in_val,out_val, in_test, out_test = trainData.GetData3(1,1,12);

    print ('in_train',in_train.shape)
    print ('out_train',out_train.shape)
    print ('in_val',in_val.shape)
    print ('out_val',out_val.shape)
    print ('in_test',in_test.shape)
    print ('out_test',out_test.shape)

    trainData.SaveAsImage(out_val, "./ROI/weights/out_val",out_val.shape[0])
    
def Test_Label():

    in_train,out_train,label_train,in_val,out_val,label_val,in_test,out_est,label_test = trainData.GetROISet(1,1,1)
    print ('train',in_train.shape)
    print ('train',out_train.shape )
    print ('train',label_train.shape )
    print ('train',label_train )

    print ('valid',in_val.shape)
    print ('valid',out_val.shape )
    print ('valid',label_val.shape )
    print ('valid',label_val )

    print ('test',in_test.shape)
    print ('test',out_est.shape )
    print ('test',label_test.shape )
    print ('test',label_test )
    print ('test',label_test[:,0] )
    print ('test flipud',1 - label_test[:,0] )

#Test_Aug()
#Test_validate()
Test_Label()