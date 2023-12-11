import joblib
import pandas as pd
import pickle

import os
os.getcwd()
dafa = pd.read_csv("TestDx.csv")

def predict(n):
    
    # dafa = pd.read_csv("TestDx.csv")
    
    with  open('ANNDanger.pkl','rb') as file:
        ann = pickle.load(file)
    # clf = joblib.load("ANNDanger.joblib")
    sampled_df = dafa.sample(n) 
    predictions = ann.predict(sampled_df)
    
    return predictions


# import joblib
# # from Models import ANNDanger.joblib as model
# import pandas as pd
# import os
# print(os.getcwd())



# df = pd.read_csv("TestDx.csv")

# def predict(n):
#     clf = joblib.load("ANNDanger.joblib")
#     df = df.sample(n)
    
#     return clf.predict(df)

# def predict(a):
#     dic = {
#         "name":"harsh",
#         "age":21,
#         "city":"noida"
#     }
    
#     return dic
    