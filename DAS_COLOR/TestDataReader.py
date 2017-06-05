from DataReader import DataReader
import numpy

trainData = DataReader();

inData,outData = trainData.GetDataAug(1,2);


print (inData.shape)
print (outData.shape)

for n in range(0, inData.shape[0]):
    print ("inData ",n, " max ",numpy.max(inData[n]), ' min ' , numpy.min(inData[n]), ' mean ',  numpy.mean(inData[n]))
    print ("outData ",n," max ", numpy.max(outData[n]), ' min ' ,numpy.min(outData[n]), ' mean ', numpy.mean(outData[n]))


trainData.SaveAsImage(outData, "./DAS_COLOR/weights/out",outData.shape[0])
#trainData.SaveAsImageByChannel(inData, "./DAS_COLOR/weights/in",)