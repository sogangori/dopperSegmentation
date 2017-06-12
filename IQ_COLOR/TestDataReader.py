from DataReader import DataReader
import numpy

trainData = DataReader();
aug = 2
count = 2
inData,outData = trainData.GetDataAug(count =count, aug = aug, isTrain = True);

print (inData.shape)
print (outData.shape)

for c in range(0, outData.shape[0]):
    print ("inData ",c, " max ",numpy.max(inData[c]), ' min ' , numpy.min(inData[c]), ' mean ',  numpy.mean(inData[c]))
    print ("outData ",c," max ", numpy.max(outData[c]), ' min ' ,numpy.min(outData[c]), ' mean ', numpy.mean(outData[c]))


#trainData.SaveAsImageByChannel(inData, "./IQ_COLOR/weights/in")
trainData.SaveAsImage(outData, "./IQ_COLOR/weights/out",outData.shape[0])