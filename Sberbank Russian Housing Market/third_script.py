
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import warnings



output = pd.read_csv('output-default.csv')
df_sub = pd.read_csv('df_sub-default.csv')
gunja_output = pd.read_csv('gunja_output-default.csv')



first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])
#first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +
#                                    .286*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2
first_result["price_doc"] = np.exp( .564*np.log(first_result.price_doc_louis) +
                                    .436*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2


result = first_result.merge(gunja_output, on="id", suffixes=['_follow','_gunja'])

result["price_doc"] = np.exp( .75*np.log(result.price_doc_follow) +
                              .25*np.log(result.price_doc_gunja) )

result["price_doc"] =result["price_doc"] *0.9900 

result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)
result.head()
result.to_csv('sub-mix-new-0.9900.csv', index=False)
0.31028

first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])
#first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +
#                                    .286*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2
first_result["price_doc"] = np.exp( .564*np.log(first_result.price_doc_louis) +
                                    .436*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2


result = first_result.merge(gunja_output, on="id", suffixes=['_follow','_gunja'])

result["price_doc"] = np.exp( .75*np.log(result.price_doc_follow) +
                              .25*np.log(result.price_doc_gunja) )

result["price_doc"] =result["price_doc"] *0.9915 

result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)
result.head()
result.to_csv('sub-mix-new-0.9915.csv', index=False)


