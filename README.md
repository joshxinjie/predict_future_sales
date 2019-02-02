# Predict Future Sales

## Description
Project for the Predict Future Sales competition at [Kaggle](https://www.kaggle.com/c/competitive-data-science-predict-future-sales). The final model is a simple stacking of 3 different models with a ridge regression model. The ridge regression model will take in the prediction outputs of three XGBoost models as inputs. The target for the ridge regression model will be the original number of items sold for each shop, item and month combination.

For example,

| A | B | C | target |
|---|---|---|--------|
|2.7|3.1|3.7|   4    |
|5.6|6.2|4.9|  5.8   |

where

A = Predictions from XGBoost Model #1
B = Predictions from XGBoost Model #2
C = Predictions from XGBoost Model #3

The three XGboost models are

1. XGBoost model with expanding mean encodings of target (item_cnt_day), item_price and revenue grouped-by shop_id and item_id. The 1-,2-,3-,4-,5-,6-,7-,8-,9-,10-,11- and 12-months lagged values of the expanding mean will also be generated. Code for this model is found in XGB_Expanding_Mean.ipynb.
2. XGBoost model with lagged mean encodings of target (item_cnt_day), item_price and revenue grouped-by (item_id, date_block_num), (shop_id, date_block_num), (item_category_id, date_block_num), (item_id, shop_id, date_block_num), (shop_id, item_category_id, date_block_num). The 1-,3-,6-,9- and 12-months lagged values of these mean encodings will be generated as well. Code for this model is found in XGB_date.ipynb.
3. Same as model #2 but the 2-,4-,5-,7- and 8-months lagged values of the mean encodings will be generated instead. Code for this model is found in XGB_date.ipynb.

The code for the ridge regression stacking model can be found in Ensemble.ipynb.
