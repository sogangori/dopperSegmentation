from DataReader import DataReader
import numpy

trainData = DataReader();

inData,outData = trainData.GetTrainDataToTensorflow(3, isTrain =   True, trainH = 64);


print (inData.shape)
print (outData.shape)

for c in range(0, outData.shape[0]):
    print ("inData ",c, " max ",numpy.max(inData[c]), ' min ' , numpy.min(inData[c]), ' mean ',  numpy.mean(inData[c]))
    print ("outData ",c," max ", numpy.max(outData[c]), ' min ' ,numpy.min(outData[c]), ' mean ', numpy.mean(outData[c]))


trainData.SaveAsImage(outData, "./DAS/weights/out",outData.shape[0])
#trainData.SaveAsImageByChannel(inData, "in")