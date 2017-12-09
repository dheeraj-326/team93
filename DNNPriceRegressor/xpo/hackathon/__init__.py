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
            print(inputParams)
            output = regressor.predict(inputParams)
            outDict = {"Total Cost" : output[0]}
            return jsonify(outDict)
        
@app.route('/test', methods=['GET'])
def test():
    dict = {'hello':'there'}
    return jsonify(dict)

# val = parser.parse('6/22/2017 20:30')
# print(val.hour)
SingletonRegressor.initialiser()
app.run(host='TVMATP379617D', port=8008, debug=True)