'''
Created on Dec 9, 2017

@author: dheeraj.b
'''
from xpo.hackathon.regressor import Regressor
from xpo.hackathon.regressor.knnregressor import KNNRegressor
class SingletonRegressor:
    regressor = None
    knregressor = None
    
    @staticmethod
    def initialiser(root_path, retrain=False):
        SingletonRegressor.regressor = Regressor(root_path, retrain)
        SingletonRegressor.knregressor = KNNRegressor(root_path)
