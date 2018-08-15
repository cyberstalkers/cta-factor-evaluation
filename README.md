# cta-factor-evaluation

CTA Stategy Factor Evaluation For Futures


### Content

This repository contains all the codes used to scrape futures trading data, conduct data preprocessing, select factors, and do backtesting.

The files are organized by:

factor_selection.py: a module that contains functions of data cleaning, preprocessing, exploratory analysis, feature engineering, training & testing, backtesting, etc.

CTA factors evaluation for iron.ipynb: a jupyter notebook file to perform factor selection for iron.


### Research Logic

1. Generate dataset including target prices and potential factors;
2. Preprocessing: 
  1) Modify data acquisition time to avoid future data; 
  2) Adjust factor update frequency to daily by forward fullfill;
  3) Correlation test;
  4) Stationarity test and seasonal adjustment;
  5) Deal with outliers.
3. Factor selection: score factors based on tertile t value.
4. Backtest: evaluated by cumulative return rate and max drawdown.
