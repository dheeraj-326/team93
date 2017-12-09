import os, sys
currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path: # add parent dir to paths
    sys.path.append(rootDir)
from flask import Flask
from flask import request
from flask.json import jsonify
from xpo.hackathon.regressor.SingletonRegressor import SingletonRegressor
import datetime
import dateutil
from dateutil import parser

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        jsonData = request.get_json(force=True)
        if jsonData is None:
            return jsonify('')
        else:
            regressor = SingletonRegressor.regressor
            pmargin = 0.15
            inputParams = addParams(jsonData)
            print(inputParams)
            output = regressor.predict(inputParams)
            output = output[0]
            outDict = {"BasePrice" : output}
            val = output + (output*pmargin)
            outDict["Price"] = val
            val = pmargin * 100
            outDict["ProfitMargin"] = val
            datesDict = {}
#             today = datetime.datetime.now()
#             for i in range(0, 5):                
#                 jsonData['PickupDate'] = today
#                 today = today + timedelta(days=1)
#                 inputParams = addParams(jsonData)
#                 regressor.predict(inputParams)
                
            return jsonify(outDict)
        
@app.route('/knpredict', methods=['POST'])
def knpredict():
        if request.method == 'POST':
            jsonData = request.get_json(force=True)
        if jsonData is None:
            return jsonify('')
        else:
            regressor = SingletonRegressor.knregressor
            pmargin = 0.15
            inputParams = addParams(jsonData)
            print(inputParams)
            output = regressor.predict(inputParams)
            output = output[0]
            outDict = {"BasePrice" : output}
            val = output + (output*pmargin)
            outDict["Price" : val]
            val = pmargin * 100
            outDict["ProfitMargin" : val]
            datesDict = {}
#             today = datetime.datetime.now()
#             for i in range(0, 5):                
#                 jsonData['PickupDate'] = today
#                 today = today + timedelta(days=1)
#                 inputParams = addParams(jsonData)
#                 regressor.predict(inputParams)
                
            return jsonify(outDict)
        
@app.route('/test', methods=['GET'])
def test():
    dict = {'hello':'there'}
    return jsonify(dict)

def addParams(jsonData):
    inputParams = {}
    orderDate = datetime.datetime.now()
    for k in jsonData:
        val = jsonData[k]
        if k != 'PickupDate':
            inputParams[k] = val
        else:
            puEarly = jsonData[k]
            puEarly = parser.parse(puEarly)
            inputParams['PUMonth'] = puEarly.month
            inputParams['PUDay'] = puEarly.day
            inputParams['PUYear'] = puEarly.year
            inputParams['PUHour'] = puEarly.hour
            inputParams['PUMinutes'] = puEarly.minute
    inputParams['OrderMonth'] = orderDate.month
    inputParams['OrderDay'] = orderDate.day
    inputParams['OrderYear'] = orderDate.year
    inputParams['OrderHour'] = orderDate.hour
    inputParams['OrderMinutes'] = orderDate.minute
    return inputParams

# val = parser.parse('6/22/2017 20:30')
# print(val.hour)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
print("here")
print(root_path)
SingletonRegressor.initialiser(root_path)
# SingletonRegressor.regressor.trainModel()
# SingletonRegressor.knregressor.trainModel()
app.run(host='localhost', port=8008, debug=False)