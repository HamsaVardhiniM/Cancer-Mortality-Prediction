import pandas as pd
class IngestData:
    def __init__(self):
        self.path=None
    def dataframe(self, path:str):
        self.path=path
        df=pd.read_csv(self.path)
        return df
