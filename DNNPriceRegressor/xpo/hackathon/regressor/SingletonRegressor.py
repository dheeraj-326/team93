'''
Created on Dec 9, 2017

@author: dheeraj.b
'''
from xpo.hackathon.regressor import Regressor
class SingletonRegressor:
    regressor = None
    
    @staticmethod
    def initialiser():
        SingletonRegressor.regressor = Regressor()
