#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App designed to show forecast scaling of a trend signal

@author: purchaja
"""
import streamlit as st
import numpy as np
import pandas as pd
import quandl
from itertools import product
import statsmodels.api as sm

#title and open quandl API 
st.title('Trend Signal Dashboard')

quandl.ApiConfig.api_key = "znv8zQ4xXa_8CS-XhQdB"
data = quandl.get("USTREASURY/YIELD").loc[:,'2 YR':'30 YR']

# choose your weapon
option = st.sidebar.selectbox(
    'Select Asset:',
     data.columns)

# EWMA sliders
EMA1 = st.sidebar.slider('Choose EMA1', 8, 60, value=8, step= 4) 
EMA2 = st.sidebar.slider('Choose EMA2', 48, 252, value=48, step=10)

# Current Signals across available 'assets'
live = pd.DataFrame()
for asset in data.columns:
    dash = pd.DataFrame(data[asset])[-2000:]
    dash.dropna(inplace = True)
    dash['EMA1'] = dash[asset].ewm(span=EMA1).mean()
    dash['EMA2'] = dash[asset].ewm(span=EMA2).mean()
    dash['Returns'] = -((dash[asset]-dash[asset].shift(1))*100)
    dash.dropna(inplace = True)
    dash['XO'] = dash['EMA2']-dash['EMA1']
    dash['Vol'] = dash['Returns'].rolling(20).std()
    dash['raw_signal'] = 10/(dash['XO'].abs().mean()+(1/10**100)) * dash['XO']
    dash['capped_signal'] = dash['raw_signal'].apply(lambda x: 20 if x > 20 else (-20 if x < -20 else x))
    cycle, trend = sm.tsa.filters.hpfilter(dash['capped_signal'], 10000)
    dash['capped_smoothed_signal'] = trend.apply(lambda x: 20 if x > 20 else (-20 if x < -20 else x))
    dash.dropna(inplace = True)
    live = live.append(pd.DataFrame(
                        {'Asset': asset,
                         'Yield': dash[asset][-1],
                         'Yield 1w chg': (dash[asset][-1]-dash[asset][-6])*100,
                         'Vol': dash['Vol'][-1],
                         'Live sig': dash['capped_smoothed_signal'][-1],
                         'Sig 1m chg': dash['capped_smoothed_signal'][-1]-dash['capped_smoothed_signal'][-30]},
                        index=[0]), ignore_index=True)

st.write(live)

# main dataframe
df = data[[option]].copy()[-2000:]
df['EMA1'] = df[option].ewm(span=EMA1).mean()
df['EMA2'] = df[option].ewm(span=EMA2).mean()
df['XO'] = df['EMA2']-df['EMA1']
df['raw_signal'] = 10/(df['XO'].abs().mean()) * df['XO']
df['capped_signal'] = df['raw_signal'].apply(lambda x: 20 if x > 20 else (-20 if x < -20 else x))
cycle, trend = sm.tsa.filters.hpfilter(df['capped_signal'], 10000)
df['capped_smoothed_signal'] = trend.apply(lambda x: 20 if x > 20 else (-20 if x < -20 else x))
    
# return functions
@st.cache
def max_drawdown(return_series):
    comp_ret = (return_series).cumsum()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret-peak)
    return dd.min()
@st.cache
def sortino_ratio(series, N):
    mean = series.mean() * N
    std_neg = series[series<0].std()*np.sqrt(N)
    return mean/std_neg
@st.cache
def sharpe_ratio(return_series, N):
    mean = return_series.mean() * N
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

#return in basis points equals the daily change * -position.
df['MKT_Return'] = ((df[option]-df[option].shift(1))*100)
df['Strat'] = df['MKT_Return']*-(df['capped_smoothed_signal']/20).shift(1)
df.dropna(inplace=True)
qmhp = pd.DataFrame({'STRAT_RETURN': df['Strat'].sum(),
                'Sharpe': sharpe_ratio(df['Strat'],252),
                'Sortino': sortino_ratio(df['Strat'],252),
                'Max Drawdown': max_drawdown(df['Strat']),
                'Start': df.first('1D').index.strftime('%d-%b-%y'),
                'End': df.last('1D').index.strftime('%d-%b-%y')},
                index=[0])
st.write('Return stats for selected asset: '+ option)
st.write(qmhp)

'Cumulative returns for selected strategy'
st.line_chart(df['Strat'].cumsum())

# EWMA optimisation
run_opt = st.checkbox('Run optimisation?')
if run_opt:
    with st.spinner('Processing...'):
        ema1 = range(8,60,4) 
        ema2 = range(48,252,10)
        results = pd.DataFrame()
        for EMA1, EMA2 in product(ema1, ema2):
            opt = pd.DataFrame(data[option])[-2000:]
            opt.dropna(inplace = True)
            opt['Returns'] = -((opt[option]-opt[option].shift(1))*100) # market returns
            opt['EMA1'] = opt[option].ewm(span=EMA1).mean()
            opt['EMA2'] = opt[option].ewm(span=EMA2).mean()
            opt.dropna(inplace = True)
            opt['XO'] = opt['EMA2']-opt['EMA1']
            opt['raw_signal'] = 10/(opt['XO'].abs().mean()+(1/10**100)) * opt['XO']
            opt['capped_signal'] = opt['raw_signal'].apply(lambda x: 20 if x > 20 else (-20 if x < -20 else x))
            cycle, trend = sm.tsa.filters.hpfilter(opt['capped_signal'], 10000)
            opt['capped_smoothed_signal'] = trend.apply(lambda x: 20 if x > 20 else (-20 if x < -20 else x))
            opt['Strategy'] = opt['Returns']*(opt['capped_smoothed_signal']/20).shift(1) # shift pos to avoid foresight bias
            opt.dropna(inplace = True)
            perf = opt[['Strategy']].sum()
            perf['Sharpe'] = sharpe_ratio(opt['Strategy'],252)
            perf['Sortino'] = sortino_ratio(opt['Strategy'], 252)
            perf['Max_DD'] = max_drawdown(opt['Strategy'])
            results = results.append(pd.DataFrame(
                        {'EMA1': EMA1, 'EMA2': EMA2, 
                        'STRAT_RETURN': perf['Strategy'],
                        'Sharpe': perf['Sharpe'],
                        'Sortino': perf['Sortino'],
                        'Max Drawdown': perf['Max_DD'],
                        'Start': opt.first('1D').index.strftime('%d-%b-%y'),
                        'End': opt.last('1D').index.strftime('%d-%b-%y')},
                        index=[0]), ignore_index=True)
        'Optimal EWMA lags sorted by Sharpe Ratio'
        st.write(results.sort_values('Sharpe', ascending=False).head())
    st.success('Optimisation Complete.  Beware of overfitting.')
   
# chart 1 
st.subheader('Yield plot with EWMA')
st.line_chart(df.loc[:, option:'EMA2'])

# chart 2 
st.subheader('Signal: +20 is max long, -20 is max short')
st.line_chart(df.loc[:, 'raw_signal':'capped_smoothed_signal'])

#Option to show dataframe
if st.checkbox('Show dataframe'):
    chart_data = df
    chart_data
    
#side notes
st.sidebar.write('This app showcases an illustrative filtered trend model. '
                 'A simple trend signal is converted to a smoothed forecast '
                 'capped at ±20. The app shows some backtesting and '
                 'optimisation of risk-adjusted returns. The process is '
                 'transferable to a broader range of assets (e.g. G10 rates, '
                 'curve and flies) and other signals.')
