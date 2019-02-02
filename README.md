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

## Files
1. EDA.ipynb: The exploratory data analysis for the project
2. Ensemble.ipynb: The code for the ridge regression stacking model
3. XGB_Expanding_Mean.ipynb: The code for model #1
4. XGB_date.ipynb: The code for model #2 and #3

## Instructions
1. Download the data files from the competition website and place them in the following structure:
```
|--- data
    |--- sales_train_v2.csv
    |--- items.csv
    |--- item_categories.csv
    |--- shops.csv
    |--- test.csv
|--- EDA.ipynb
|--- Ensemble.ipynb
|--- XGB_Expanding_Mean.ipynb
|--- XGB_date.ipynb
```
2. Run EDA.ipynb if you are interested in the exploratory data analysis
3. Run XGB_Expanding_Mean.ipynb. This will generate 4 files. xgb_expmean_submission.csv is the test set predictions from this model and you can submit it to Kaggle to see the performance of only this model. xgb_expmean_train.csv and xgb_expmean_valid.csv will be used for the model stacking. xgb_expmean.pickle.dat is the saved trained model
4. Run XGB_date.ipynb. In the second cell, uncomment ```lags=[1,3,6,9,12]``` and comment ```lags=[2,4,5,7,8]```. This will run model #2. 4 files will be generated. xgb_date_1_3_6_9_12_submission.csv is the test set prediction for this model and you can submit it to Kaggle to see the performance of only this model. xgb_date_1_3_6_9_12_train.csv and xgb_date_1_3_6_9_12_valid.csv will be used for the the model stacking. xgb_date_1_3_6_9_12.pickle.dat is the saved trained model.
5. Run XGB_date.ipynb. In the second cell, comment ```lags=[1,3,6,9,12]``` and uncomment ```lags=[2,4,5,7,8]```. This will run model #3. 4 files will be generated. xgb_date_2_4_5_7_8_submission.csv is the test set prediction for this model and you can submit it to Kaggle to see the performance of only this model. xgb_date_2_4_5_7_8_train.csv and xgb_date_2_4_5_7_8_valid.csv will be used for the the model stacking. xgb_date_2_4_5_7_8.pickle.dat is the saved trained model.
6. Run Ensemble.ipynb to run the model stacking. This will generate the final_submission.csv which is the test set predictions from the model stacking. Submit this file to Kaggle to obtain the final results of the ensemble model.

## Installations
Anaconda, XGBoost, pickle
