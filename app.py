# Dataset: https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

from project.preprocessing import get_data, asel_mebaysan_preprocess
from project.model import random_forest_classifier, xgbm_classifier
from project.prediction import predict, get_test_data
from helpers.model_evaluation import plot_importance

#################################
#### * Train ####
#################################

df = get_data()
df = asel_mebaysan_preprocess(df, 'group')
X = df.drop('satisfaction_satisfied', axis=1)
y = df['satisfaction_satisfied']

result_dict_rf = random_forest_classifier(X, y)
result_dict_xgbm = xgbm_classifier(X, y)

#################################
#### * Test ####
#################################

test_df, test_X, test_y = get_test_data(na='group')

test_dict_rf = predict(test_X, test_y, result_dict_rf['model'])
test_dict_xgbm = predict(test_X, test_y, result_dict_xgbm['model'])


#################################
#### * Plot Importance for Random Forest ####
#################################
plot_importance(test_dict_rf['model'],test_X,num=15)

#################################
#### * Plot Importance for XGBM ####
#################################
plot_importance(test_dict_xgbm['model'],test_X,num=15)