import tensorflow as tf
from tensorflow.contrib.learn import DNNRegressor
# from tensorflow.python.ops.io_ops import TextLineReader
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
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

nonNumericCols = ["Equipment Type","Order Type", "Lane", "Customer Id"]
labelCol = "Total Cost USD"
# dateCols = ["Month", "Day", "UnixTime"]
dateCols = ["Month"]
inputParams = ["PickupDate","Origin Zip Code","Destination Zip Code","Distance","Loads","Lanes with Multiple Carriers","Same Day Loads","Equipment Type"]


#     print(encDf.iloc[70])
#         if i%10 == 0:
            
#         i+=1



class Regressor:
    def __init__(self):
        self.data_file_name = "CleanedDataTest.csv"
        self.model_file_name = "RandomForestPickle.pkl"
        self.featuresDFfile_name = "featuresDf.pkl"
        self.labelsfile_name = "labelsDf.pkl"
        self.nonNumEncDffile_name = "nonNumEncDf.pkl"
        
        self.nonNumEncDf = None
        self.regressor = None
        self.featuresDf = None
        self.labelsDf = None
        
        if not self.modelTrained():
#             pass
            self.trainModel()
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
        x_train, x_test, y_train, y_test =  train_test_split(finalDf, labelsDf, test_size=0.10, random_state=4)
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
        x_train = x_train.as_matrix()
        x_test = x_test.as_matrix()
#         y_train = y_train.tolist()
        feature_columns = [tf.feature_column.numeric_column('x', shape=x_train.shape[1:])]
        regressor = tf.estimator.DNNRegressor(
            feature_columns=feature_columns, hidden_units=[50, 30, 10])
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x_train}, y=y_train, batch_size=1, num_epochs=None, shuffle=True)
        regressor.train(input_fn=train_input_fn, steps=5000)
#         regressor = RandomForestRegressor(n_estimators=10000)
        
#         regressor.fit(x_train, y_train)
#         regressor.fit(x_train, y_train)
#         scaler = preprocessing.StandardScaler()
#         x_transformed = scaler.transform(x_test)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
              x={'x': x_test}, y=y_test, num_epochs=1, shuffle=False)
#         
        preds = regressor.predict(x_test)
#         plt.scatter(y_test, preds, color='black')
#         plt.plot(y_test, preds, color='blue', linewidth=3)
#         plt.xticks(())
#         plt.yticks(())
#         plt.show()
#         print(type(preds))
        for p in preds:
            print(p['predictions'])
#         y_predicted = np.array(list(p['predictions'] for p in preds))
        y_predicted = y_predicted.reshape(np.array(y_test).shape)
#         score_sklearn = metrics.mean_squared_error(preds, y_test)
        train_preds = regressor.predict(x_train)
        for p in train_preds:
            print(p['predictions'])
#         train_predicted = np.array(list(p['predictions'] for p in train_preds))
        train_predicted = y_predicted.reshape(np.array(y_test).shape)
        score_sklearn = metrics.mean_squared_error(y_predicted, y_test)
        train_score = metrics.mean_squared_error(train_predicted, y_train)     
        for i in range(len(train_preds)):
            diff = math.fabs(y_train[i]-train_preds[i])
            if diff>200:
                print("Actual ", y_train[i], "Pred ", train_preds[i], "Diff ", diff)
        print("M.S Error (Test)", score_sklearn)
        print("M.S Error (Train)", train_score)
#         regressor.predict(x_test)
#         zeros = 0
#         diffs = y_test-preds
#         sums = ((y_test-preds))/(y_test)
#         sums = sums**2
#         for diff in diffs:
#             if diff < 0.001:
#                 zeros+=1
        #     else:
        #         print(diff)        
#         for i in range(len(preds)):
#             print(preds)
#         sum = 0
#         for i in range(len(preds)):
#             n = y_test[i] - preds[i]      
# #             n = n**2 / (y_test[i])**2
#             if i % 20 == 0:
#                 print("actual ", y_test[i], "pred ", preds[i], "diff ", n)
# #                       , "x_test", x_test.iloc[i])
#             sum += n
#         sum = sum/len(preds)
#         print(sum)
#         sum = 100 - np.sum(sums)/len(preds)*100
#         print("Accuracy ", sum)
#         with open(self.model_file_name, mode="wb+") as file:
#             pickle._dump(regressor, file)
#         with open(self.labelsfile_name, mode="wb+") as file:
#             pickle._dump(labelsDf, file)
#         with open(self.featuresDFfile_name, mode="wb+") as file:
#             pickle._dump(featureDf, file)
#         with open(self.nonNumEncDffile_name, mode="wb+") as file:
#             pickle._dump(nonNumEncDf, file)
    
    def predict(self, paramsDict):
        valarr = self.encodeSingle(paramsDict)
        outputVal = self.regressor.predict(valarr)
        return outputVal
    
    def encodeSingle(self, paramsDict):
        arr = None
        dt = paramsDict['PickupDate']
        unixTime = time.mktime(dt.timetuple())
        mnth = dt.month
        year = dt.year
        keyslist = self.featuresDf.keys() 
        sz = len(keyslist)
        for k in keyslist:
            arr = np.zeros(sz).tolist()
            if k in dateCols:
                ind = keyslist.index(k)
                if k == "UnixTime":
                    arr[ind] = unixTime
                elif k == "Month":
                    arr[ind] = mnth
                else:
                    arr[ind] = year
            elif k in nonNumericCols:
                curr = paramsDict[k]
                if curr in keyslist:
                    ind = keyslist.index(curr)
                    arr[ind] = 1
            else:
                ind = keyslist.index(k)
                curr = paramsDict[k]
                arr[ind] = curr
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
    
#     def encodeDateDfBatch(self, dateDf):
#         srs = pd.Series(dateDf.stack())
#         dateDf = pd.to_datetime(srs)
#         dateList = []
#         dataSecKeys = ["Year"]
#         for i in range(0,7):
#             klabel = "Group"+str(i)
#             dataSecKeys.append(klabel)
#         i=0
#         print(type(dateDf.iloc[0]))
#         for item in dateDf:        
#             encdates = [0,0,0,0,0,0,0,0]
#             if isinstance(item, pd.Timestamp):
#                 itemDt = item.date() 
#                 year = itemDt.year            
#                 monthsect = int((itemDt.month + 1)/2)
#     #             print(i, " ", itemDt.month, " ", monthsect)
#                 encdates[0] = year
#                 encdates[monthsect+1] = 1
#     #         print(i, " ",itemDt.month, " ", monthsect)
#             dateList.append(encdates)
#             i+=1
#         encDf = pd.DataFrame.from_records(dateList, columns=dataSecKeys)
#         return encDf
    
    def modelTrained(self):
        exists = False
        try:
            a = open(self.encDateDffile_name)
            b = open(self.featuresDFfile_name)
            c = open(self.labelsfile_name)
            d = open(self.nonNumEncDffile_name)
            e = open(self.model_file_name)
            a.close()
            b.close()
            c.close()
            d.close()
            e.close()
            exists = True
        except:
            exists = False
        return exists

regress = Regressor()
regress.trainModel()
# contin = True
# while(contin):
#     pdict = {}
#     for val in inputParams:
#         inp = input(val)
#         pdict[val] = inp
#     outval = regress.predict(pdict)
#     print("Output ", outval)

# for i in sums:
#     sum += i    
# print(100 - sum*100)
# print(sum)    
# scores = (y_test - preds)/(y_test)
# print(scores)
# for itm in x_train:
#     print(itm)
# print(x_test[5])
# for itm in x_train:
#     print(itm)
# for i in range(0, len(y_test)):
#     print(x_test[i])
#     print("Pred : ", regressor.predict(x_test[i]), "Actual : ", y_test[i])


# print()
# print(finalDf)
# print(arr[0])
# print(numericDf.keys())
# print(nonNumericDf.keys())
# vls = list('abca')
# sers = pd.Series(vals)
# dummies = pd.get_dummies(sers)
# print(dummies['XPOCHA'])
# print(len(dummies))
# for i in range(0, len(dummies)):
# for val in dummies:
#     print(val, " ", dummies[val])