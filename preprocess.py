import pandas as pd
import numpy as np
import gzip
import os
from sklearn.preprocessing import MinMaxScaler
import copy
import matplotlib.pyplot as plt
def load_mnist_data(xpath,ypath,num_images):
    f = gzip.open(os.path.join(os.path.dirname(__file__),xpath), 'r')
    image_size = 28

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    data=data/255

    f = gzip.open(os.path.join(os.path.dirname(__file__),ypath), 'r')
    f.read(8)
    buf = f.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    # print(data.shape)
    # image = np.asarray(data[-1]).squeeze()
    # print(len(image))
    # plt.imshow(image)
    # plt.show()

    # print(labels.shape)
    # for i in range(0,50):
    #     buf = f.read(1)
    #     labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    #     # print(labels)

    return data,labels

def get_mnist_dataset():
    #
    #
    trainx,trainy=load_mnist_data(xpath='test_train/train-images-idx3-ubyte.gz',ypath='test_train/train-labels-idx1-ubyte.gz',num_images=30000)
    testx, testy = load_mnist_data(xpath='test_train/t10k-images-idx3-ubyte.gz',ypath='test_train/t10k-labels-idx1-ubyte.gz',num_images=10000)

    np.save('mnist_dataset/trainx.npy', trainx)
    np.save('mnist_dataset/trainy.npy', trainy)
    np.save('mnist_dataset/testx.npy', testx)
    np.save('mnist_dataset/testy.npy', testy)

def get_adult_dataset():
    dfDataTrain = pd.read_excel(os.path.join(os.path.dirname(__file__),'test_train/adult_train.xlsx'))
    dfDataTest = pd.read_excel(os.path.join(os.path.dirname(__file__), "test_train/adult_test.xlsx"))
    intTrainSize = len(dfDataTrain)
    intTestSize = len(dfDataTest)
    # print(dfDataTrain.keys())
    dfDataTrainY = dfDataTrain["income"]
    # print(type(dfDataTrainY))
    dfTrainY = pd.DataFrame((dfDataTrainY == " >50K").astype("int64"), columns=["income"])  # >50K 1, =<50K 0
    dfDataTestY = dfDataTest["income"]
    dfTestY = pd.DataFrame((dfDataTestY == " >50K.").astype("int64"), columns=["income"])  # >50K 1, =<50K 0
    dfDataTrain = dfDataTrain.drop(["income"], axis=1)
    dfDataTest = dfDataTest.drop(["income"], axis=1)

    dfAllData = pd.concat([dfDataTrain, dfDataTest], axis=0, ignore_index=True)
    # print(dfAllData)
    dfAllData=makeDataProcessing(dfData=dfAllData)
    # sperate All data to Training and Testing

    dfTrainX = dfAllData[0:intTrainSize]
    dfTestX = dfAllData[intTrainSize:(intTrainSize + intTestSize)]

    # save Training data, Testing data and Training label
    dfTrainX.to_csv(os.path.join(os.path.dirname(__file__), "adult_dataset/X_train_my.csv"), index=False)
    dfTestX.to_csv(os.path.join(os.path.dirname(__file__), "adult_dataset/X_Test_my.csv"), index=False)
    dfTrainY.to_csv(os.path.join(os.path.dirname(__file__), "adult_dataset/Y_train_my.csv"), index=False)
    dfTestY.to_csv(os.path.join(os.path.dirname(__file__), "adult_dataset/Y_Test_my.csv"), index=False)
    arrayTrainX = np.array(dfTrainX.values)
    arrayTestX = np.array(dfTestX.values)
    arrayTrainY = np.array(dfTrainY.values)
    arrayTestY= np.array(dfTestY.values)
    np.save(os.path.join(os.path.dirname(__file__), "adult_dataset/trainx.npy"),arrayTrainX)
    np.save(os.path.join(os.path.dirname(__file__), "adult_dataset/trainy.npy"), arrayTrainY)
    np.save(os.path.join(os.path.dirname(__file__), "adult_dataset/testx.npy"),arrayTestX)
    np.save(os.path.join(os.path.dirname(__file__), "adult_dataset/testy.npy"), arrayTestY)
def makeDataProcessing(dfData):
    dfDataX = dfData.drop(["educationnum","sex"], axis=1)


    listObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes=="object"]
    listNonObjectColumnName = [col for col in dfDataX.columns if dfDataX[col].dtypes!="object"]


    dfNonObjectData = dfDataX[listNonObjectColumnName]
    print(dfNonObjectData)
    normalize_dict={}
    for index,keys in enumerate(dfNonObjectData.keys()):
        print(keys)
        print(type(dfNonObjectData))
        deal_arrary=dfNonObjectData[keys].values.reshape(-1,1)
        normalize_data= np.squeeze(MinMaxScaler().fit_transform(deal_arrary))
        normalize_dict[str(keys)]=normalize_data
    dfdfNonObjectData_Normallize = pd.DataFrame(data=normalize_dict)
    dfdfNonObjectData_Normallize.insert(2, "sex", (dfData["sex"]==" Male").astype(np.int)) # Male 1 Femal 0


    dfObjectData = dfDataX[listObjectColumnName]
    dfObjectData = pd.get_dummies(dfObjectData)

    dfDataX = dfdfNonObjectData_Normallize.join(dfObjectData)
    dfDataX = dfDataX.astype("float")
    return dfDataX
def get_bike_dataset():
    dfData= pd.read_csv(os.path.join(os.path.dirname(__file__),'test_train/day.csv'))
    intDataSize = len(dfData)
    # print(intDataSize)
    intTrainxSize=int(intDataSize*0.8)
    dfDataY=dfData['cnt']
    dfDataX=dfData.drop(['cnt','casual','registered','instant','dteday'],axis=1)
    # dfAllData=makeDataProcessing(dfData=dfAllData)
    # sperate All data to Training and Testing

    dfTrainX = dfDataX[0:intTrainxSize]
    dfTestX = dfDataX[intTrainxSize:intDataSize]
    dfTrainY = dfDataY[0:intTrainxSize]
    dfTestY = dfDataY[intTrainxSize:intDataSize]

    # save Training data, Testing data and Training label
    dfTrainX.to_csv(os.path.join(os.path.dirname(__file__), "bike_dataset/X_train_my.csv"), index=False)
    dfTestX.to_csv(os.path.join(os.path.dirname(__file__), "bike_dataset/X_Test_my.csv"), index=False)
    dfTrainY.to_csv(os.path.join(os.path.dirname(__file__), "bike_dataset/Y_train_my.csv"), index=False)
    dfTestY.to_csv(os.path.join(os.path.dirname(__file__), "bike_dataset/Y_Test_my.csv"), index=False)
    arrayTrainX = np.array(dfTrainX.values)
    arrayTestX = np.array(dfTestX.values)
    arrayTrainY =np.array(dfTrainY.values.astype(int)).reshape(-1,1)
    arrayTestY= np.array(dfTestY.values.astype(int)).reshape(-1,1)
    arrayTestY=MinMaxScaler().fit_transform(arrayTestY)
    arrayTrainY=MinMaxScaler().fit_transform(arrayTrainY)
    # print(arrayTrainY)
    # arrayTrainY=sklearn.preprocessing.maxabs_scale(arrayTrainY,axis=0,copy=True)
    np.save(os.path.join(os.path.dirname(__file__), "bike_dataset/trainx.npy"),arrayTrainX)
    np.save(os.path.join(os.path.dirname(__file__), "bike_dataset/trainy.npy"), arrayTrainY)
    np.save(os.path.join(os.path.dirname(__file__), "bike_dataset/testx.npy"),arrayTestX)
    np.save(os.path.join(os.path.dirname(__file__), "bike_dataset/testy.npy"), arrayTestY)

if __name__ == "__main__":
    pass
get_adult_dataset()
# get_mnist_dataset()
# get_bike_dataset()
# train=np.load('adult_dataset/testy.npy')
# print(np.max(train))
