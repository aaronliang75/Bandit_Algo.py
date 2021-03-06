import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.stats import kstest

"""
Step 1:
Test Distribution of Stock Returns
"""
stock_data = pd.read_csv('./RL_A1_Data/hs300_daily.csv', index_col=0)

# Testing for t distribution
np.random.seed(6010)

t_testing_df = pd.DataFrame(index=stock_data.columns)
for stock in tqdm(stock_data.columns):
    return_series = stock_data[stock].dropna().values
    # student-t test
    fitted_t = stats.t.fit(return_series)
    ks_stat, p_val = kstest(return_series, 'norm')
    # params
    df = fitted_t[0]
    loc = fitted_t[1]
    scale = fitted_t[2]
    t_est = stats.t.rvs(df=df, loc=loc, scale=scale, size=len(return_series))
    t_stat, t_p_val = stats.ks_2samp(return_series, t_est)
    t_testing_df.loc[stock, 'ttest_p_value'] = t_p_val

print('The number of stocks following t-distribution: ', t_testing_df[t_testing_df['ttest_p_value'] > 0.01].shape[0])
print('Ratio = ', t_testing_df[t_testing_df['ttest_p_value'] > 0.01].shape[0] / stock_data.shape[1] * 100, ' %')

"""
Step 2:  
Stock returns roughly follow t-distribution
    Store params for t-distribution
"""

stock_t_param = pd.DataFrame(index=stock_data.columns)

for stock in tqdm(stock_data.columns):
    return_series = stock_data[stock].dropna().values
    # t test
    fitted_t = stats.t.fit(return_series)
    # params
    df = fitted_t[0]
    loc = fitted_t[1]
    scale = fitted_t[2]
    # store
    stock_t_param.loc[stock, 't_loc'] = loc
    stock_t_param.loc[stock, 't_scale'] = scale
    stock_t_param.loc[stock, 't_df'] = df

stock_t_param.to_csv('./RL_A1_Data/hs300_stock_t_params.csv')
