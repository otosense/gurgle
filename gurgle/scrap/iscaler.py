from sklearn.preprocessing import StandardScaler
from gurgle.base import CallableModel




class StdLessStandardScaler(StandardScaler):
    def __init__(self):
        super().__init__(with_std=True)


CallableModel(StdLessStandardScaler)
