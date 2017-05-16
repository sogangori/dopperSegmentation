from DataReader import DataReader
import numpy

trainData = DataReader();
aug = 1
inData,outData,helpData = trainData.GetDataROI(count = 1, aug = aug, isTrain = True);

print (inData.shape)
print (outData.shape)

for c in range(0, outData.shape[0]):
    print ("inData ",c, " max ",numpy.max(inData[c]), ' min ' , numpy.min(inData[c]), ' mean ',  numpy.mean(inData[c]))
    print ("outData ",c," max ", numpy.max(outData[c]), ' min ' ,numpy.min(outData[c]), ' mean ', numpy.mean(outData[c]))


#trainData.SaveAsImageByChannel(inData, "./Color/weights/in")
trainData.SaveAsImage(outData, "./Color/weights/out",outData.shape[0],aug)
trainData.SaveAsImage(helpData, "./Color/weights/help",helpData.shape[0])