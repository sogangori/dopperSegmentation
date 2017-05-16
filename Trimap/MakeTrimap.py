from DataReader import DataReader
import numpy

trainData = DataReader();

inData,outData = trainData.GetTrainDataToTensorflowTest(isTrain =   True);


print ("inData",inData.shape)
print ("outData",outData.shape)

for c in range(0, outData.shape[0]):
    print ("inData ",c, " max ",numpy.max(inData[c]), ' min ' , numpy.min(inData[c]), ' mean ',  numpy.mean(inData[c]))
    print ("outData ",c," max ", numpy.max(outData[c]), ' min ' ,numpy.min(outData[c]), ' mean ', numpy.mean(outData[c]))


trainData.SaveAsImage(outData, "out")
trainData.SaveAsImageByChannel(inData, "in")