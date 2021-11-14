# Dataset: https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

from project.preprocessing import asel_mebaysan_preprocess
from project.model import random_forest_classifier, xgbm_classifier

df = asel_mebaysan_preprocess()

df.head()

X = df.drop('satisfaction_satisfied', axis=1)
y = df['satisfaction_satisfied']

xgbm_classifier(X, y)
random_forest_classifier(X, y)