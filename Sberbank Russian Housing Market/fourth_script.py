################################################################################################
################################################################################################
################################################################################################
# Combine the results
# Last model is based on a legitimate data cleaning,
#      but there's also a small transformation applied ot the predictions,
#      so also probably not generalizable),
################################################################################################
################################################################################################
################################################################################################

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
from scipy.stats import norm

magic= pd.read_csv('output1.csv')
sub_mix_new_09900= pd.read_csv('sub-mix-new-0.9900.csv')
sub_mix_new_09915= pd.read_csv('sub-mix-new-0.9915.csv')

first_result = sub_mix_new_09900.merge(magic, on="id", suffixes=['_louis','_bruno'])
#first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +
#                                    .286*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2
first_result["price_doc"] = np.exp( .8*np.log(first_result.price_doc_louis) +
                                    .2*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2

# first_result.head()
# first_result.drop(["price_doc_louis","price_doc_bruno"],axis=1,inplace=True)
# first_result.head()
# first_result.to_csv('sub-mix-output-df_sub.csv', index=False)
result = first_result.merge(sub_mix_new_09915, on="id", suffixes=['_follow','_gunja'])

result["price_doc"] = np.exp( .58*np.log(result.price_doc_follow) +
                              .42*np.log(result.price_doc_gunja) )

result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)
result.head()
result.to_csv('sub-mix-4 3-3.csv', index=False)

preds = result
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Parameters
prediction_stderr = 0.025  #  assumed standard error of predictions
                          #  (smaller values make output closer to input)
train_test_logmean_diff = 0.1  # assumed shift used to adjust frequencies for time trend
probthresh = 80  # minimum probability*frequency to use new price instead of just rounding
rounder = 4  # number of places left of decimal point to zero

# Select investment sales from training set and generate frequency distribution
invest = train[train.product_type=="Investment"]
freqs = invest.price_doc.value_counts().sort_index()

# Select investment sales from test set predictions
test_invest_ids = test[test.product_type=="Investment"]["id"]
invest_preds = pd.DataFrame(test_invest_ids).merge(preds, on="id")

# Express X-axis of training set frequency distribution as logarithms, 
#    and save standard deviation to help adjust frequencies for time trend.
lnp = np.log(invest.price_doc)
stderr = lnp.std()
lfreqs = lnp.value_counts().sort_index()

# Adjust frequencies for time trend
lnp_diff = train_test_logmean_diff
lnp_mean = lnp.mean()
lnp_newmean = lnp_mean + lnp_diff

def norm_diff(value):
    return norm.pdf((value-lnp_diff)/stderr) / norm.pdf(value/stderr)

newfreqs = lfreqs * (pd.Series(lfreqs.index.values-lnp_newmean).apply(norm_diff).values)

# Logs of model-predicted prices
lnpred = np.log(invest_preds.price_doc)

# Create assumed probability distributions
stderr = prediction_stderr
mat =(np.array(newfreqs.index.values)[:,np.newaxis] - np.array(lnpred)[np.newaxis,:])/stderr
modelprobs = norm.pdf(mat)

# Multiply by frequency distribution.
freqprobs = pd.DataFrame( np.multiply( np.transpose(modelprobs), newfreqs.values ) )
freqprobs.index = invest_preds.price_doc.values
freqprobs.columns = freqs.index.values.tolist()

# Find mode for each case.
prices = freqprobs.idxmax(axis=1)

# Apply threshold to exclude low-confidence cases from recoding
priceprobs = freqprobs.max(axis=1)
mask = priceprobs<probthresh
prices[mask] = np.round(prices[mask].index,-rounder)

# Data frame with new predicitons
newpricedf = pd.DataFrame( {"id":test_invest_ids.values, "price_doc":prices} )
newpricedf["price_doc"] =newpricedf["price_doc"] *1.0020  
# Merge these new predictions (for just investment properties) back into the full prediction set.
newpreds = preds.merge(newpricedf, on="id", how="left", suffixes=("_old",""))
newpreds.loc[newpreds.price_doc.isnull(),"price_doc"] = newpreds.price_doc_old
newpreds.drop("price_doc_old",axis=1,inplace=True)
newpreds.head()

newpreds.to_csv('final_result.csv', index=False)
3.1017
