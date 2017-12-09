# from tensorflow.python.ops.io_ops import TextLineReader
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from numpy import array
import math
import pickle
import matplotlib.pyplot as plt
import time
import dateutil.parser
from sklearn import metrics
from sklearn import preprocessing
import traceback
import os

nonNumericCols = ["Equipment Type","Order Type", "Customer Id"]
labelCol = "Total Cost USD"
# dateCols = ["Month", "Day", "UnixTime"]
dateCols = ["Month"]
inputParams = ["PickupDate","Origin Zip Code","Destination Zip Code","Distance","Loads","Lanes with Multiple Carriers","Same Day Loads","Equipment Type"]


#     print(encDf.iloc[70])
#         if i%10 == 0:
            
#         i+=1



class KNNRegressor:
    def __init__(self, root_path, retrain = False):
        self.data_file_name = "CleanedData.csv"
        self.data_file_name = os.path.join(root_path, "hackathon", "regressor", "CleanedData.csv")
        self.model_file_name = os.path.join(root_path, "hackathon", "regressor" "KNNPickle.pkl")
        self.finalDFfile_name = os.path.join(root_path, "hackathon", "regressor" "knnfinalDf.pkl")
        self.labelsfile_name = os.path.join(root_path, "hackathon", "regressor" "knnlabelsDf.pkl")
        self.nonNumEncDffile_name = os.path.join(root_path, "hackathon", "regressor" "knnnonNumEncDf.pkl")
        
        self.nonNumEncDf = None
        self.regressor = None
        self.finalDf = None
        self.labelsDf = None
        print(self.modelTrained())
        if not self.modelTrained():
#             pass
            self.trainModel()
            self.modelTrained()
        else:
            print("Saved model found")
    
    def trainModel(self):
        data = pd.read_csv(self.data_file_name)
        featureCols = list(data.keys())
        print(featureCols)
        featureCols.remove(labelCol)
        featureDf = data.filter(items=featureCols)
        labelsDf = data.filter(items=[labelCol])
        labelsDf = labelsDf.fillna(0)
        
        numericDf = featureDf._get_numeric_data()
        print(numericDf.keys())
        nonNumericDf = featureDf.filter(items=nonNumericCols)
        finalDf, nonNumEncDf = self.getEncodedBatch(numericDf, nonNumericDf)                
#         regressor = KNeighborsRegressor()
        finalDf = finalDf.fillna(0)
#         f = open("samplefile.txt", "w")
#         for
#         print(finalDf.as_matrix())
#         for ele in finalDf.as_matrix():
#             f.write(str(ele))
#         f.close()
        x_train, x_test, y_train, y_test =  train_test_split(finalDf, labelsDf, test_size=0.30, random_state=4)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
#         x_train_dict = {}
#         x_test_dict = {}
#         x_train = np.array(x_train.a)
#         x_test = np.array(x_test.as_matrix())
#         for itm in x_train:
#             x_train_dict[itm] = x_train[itm]
#         for itm in x_test:
#             x_test_dict[itm] = x_test[itm] 
#         [x_train_dict[k] = x_train[k] for k in x_train]
        print(x_train.shape)
        print(y_train.shape)
#         x_train = x_train.as_matrix()
#         x_test = x_test.as_matrix()
#         y_train = y_train.tolist()
#         feature_columns = [tf.feature_column.numeric_column('x', shape=x_train.shape[1:])]
#         regressor = tf.estimator.DNNRegressor(
#             feature_columns=feature_columns, hidden_units=[50, 30, 10])
#         train_input_fn = tf.estimator.inputs.numpy_input_fn(
#             x={'x': x_train}, y=y_train, batch_size=1, num_epochs=None, shuffle=True)
#         regressor.train(input_fn=train_input_fn, steps=5000)
        regressor = KNeighborsRegressor(n_neighbors=4)
        
        regressor.fit(x_train, y_train)
#         regressor.fit(x_train, y_train)
#         scaler = preprocessing.StandardScaler()
#         x_transformed = scaler.transform(x_test)
#         test_input_fn = tf.estimator.inputs.numpy_input_fn(
#               x={'x': x_test}, y=y_test, num_epochs=1, shuffle=False)
#         
        preds = regressor.predict(x_test)
#         plt.scatter(y_test, preds, color='black')
#         plt.plot(y_test, preds, color='blue', linewidth=3)
#         plt.xticks(())
#         plt.yticks(())
#         plt.show()
#         print(type(preds))
#         for p in preds:
#             print(p['predictions'])
#         y_predicted = np.array(list(p['predictions'] for p in preds))
#         y_predicted = y_predicted.reshape(np.array(y_test).shape)
#         score_sklearn = metrics.mean_squared_error(preds, y_test)
        train_preds = regressor.predict(x_train)
#         for p in train_preds:
#             print(p['predictions'])
#         train_predicted = np.array(list(p['predictions'] for p in train_preds))
#         train_predicted = y_predicted.reshape(np.array(y_test).shape)
        score_sklearn = metrics.mean_squared_error(preds, y_test)
        train_score = metrics.mean_squared_error(train_preds, y_train)     
#         for i in range(len(train_preds)):
#             diff = math.fabs(y_train[i]-train_preds[i])
#             if diff>200:
#                 print("Actual ", y_train[i], "Pred ", train_preds[i], "Diff ", diff)
        print("M.S Error (Test)", score_sklearn)
        print("M.S Error (Train)", train_score)
        
        with open(self.model_file_name, mode="wb") as file:
            pickle._dump(regressor, file)
        with open(self.labelsfile_name, mode="wb") as file:
#             mat = labelsDf.as_matrix()
            pickle._dump(labelsDf, file)
        with open(self.finalDFfile_name, mode="wb") as file:
#             mat = featureDf.as_matrix()
            pickle._dump(finalDf, file)
        with open(self.nonNumEncDffile_name, mode="wb") as file:
#             mat = nonNumEncDf.as_matrix()
            pickle._dump(nonNumEncDf, file)
        print("Saved")
    
    def predict(self, paramsDict):
        valarr = np.array(self.encodeSingle(paramsDict))
        valarr = valarr.reshape(1, -1)
        outputVal = self.regressor.predict(valarr)
        return outputVal
    
    def encodeSingle(self, paramsDict):
        arr = None
        keyslist = list(self.finalDf.keys())
        paramKeys = list(paramsDict.keys())
        print(keyslist)
        sz = len(keyslist)
        arr = np.zeros(sz).tolist()
        for k in paramKeys:            
#             if k in dateCols:
#                 ind = keyslist.index(k)
#                 if k == "UnixTime":
#                     arr[ind] = unixTime
#                 elif k == "Month":
#                     arr[ind] = mnth
#                 else:
#                     arr[ind] = year
            if k in keyslist:
                curr = paramsDict[k]
                ind = keyslist.index(k)
                arr[ind] = curr
                
            else:
                curr = paramsDict[k]
                if curr in keyslist:
                    ind = keyslist.index(curr)
                    arr[ind] = 1
#             vals.append(arr)
        return arr
    
    
    def getEncodedBatch(self, numericDf, nonNumericDf):
        feature_dict = {}
        numericList = [numericDf]
        dfList = [numericDf]
        nonNumEncDfs = []
        for frameKey in nonNumericDf:
            values = nonNumericDf[frameKey]
            dataSeries = pd.Series(values)
            dummies = pd.get_dummies(dataSeries)
            nonNumEncDfs.append(dummies)
        [dfList.append(itm) for itm in nonNumEncDfs]
        finalFrame = pd.concat(dfList, axis=1)
        nonNumEncDf = None
        if len(nonNumEncDfs)>0:
            nonNumEncDf = pd.concat(nonNumEncDfs, axis=1)
        return finalFrame, nonNumEncDf
    
    
    def modelTrained(self):
        exists = False
        try:
            b = open(self.finalDFfile_name, mode="rb")
#             mat = pickle.load(b)
#             self.finalDf = pd.DataFrame(mat)
            self.finalDf = pickle.load(b)
            c = open(self.labelsfile_name, mode="rb")
#             mat = pickle.load(c)
#             self.labelsDf = pd.DataFrame(mat)
            self.labelsDf = pickle.load(c)
            d = open(self.nonNumEncDffile_name, mode="rb")
#             mat = pickle.load(d)
#             self.nonNumEncDf = pd.DataFrame(mat)
            self.nonNumEncDf = pickle.load(d)
            e = open(self.model_file_name, mode="rb")
            self.regressor = pickle.load(e)
            b.close()
            c.close()
            d.close()
            e.close()
            exists = True
        except Exception as e:
            print(e)
            traceback.print_exc()
            exists = False
        return exists

# regress = KNNRegressor(retrain=True)
# regress.trainModel()
