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
        
    trainData.SaveAsImage(outData, "./DAS_Unknown/weights/out",outData.shape[0])
    #trainData.SaveAsImageByChannel(inData, "./DAS_COLOR/weights/in",)

def Test_validate():
    in_train,out_train,in_val,out_val, in_test, out_test = trainData.GetData3(24,1,12);

    print ('in_train',in_train.shape)
    print ('out_train',out_train.shape)
    print ('in_val',in_val.shape)
    print ('out_val',out_val.shape)
    print ('in_test',in_test.shape)
    print ('out_test',out_test.shape)

    trainData.SaveAsImage(out_val, "./DAS_Unknown/weights/out_val",out_val.shape[0])
    

#Test_Aug()
Test_validate()