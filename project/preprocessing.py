import pandas as pd
import numpy as np
from helpers.eda import grab_col_names, check_df, high_correlated_cols, cat_summary, num_summary

pd.set_option('display.max_columns', None)

def get_data():
    return pd.read_csv('./data/train.csv').drop('Unnamed: 0', axis=1)


def age_engineering(df):
    # ERGEN ERKEK - KADIN 0-17
    df.loc[(df["Age"] <= 17) & (df["Gender"] == "Male"),
           "NEW_GENDER_CAT"] = "AdolescentMale"
    df.loc[(df["Age"] <= 17) & (df["Gender"] == "Female"),
           "NEW_GENDER_CAT"] = "AdolescentFemale"
    # GENÇ ERKEK-KADIN 18-65
    df.loc[(df["Age"] >= 18) & (df["Age"] <= 65) & (
        df["Gender"] == "Male"), "NEW_GENDER_CAT"] = "YouthMale"
    df.loc[(df["Age"] >= 18) & (df["Age"] <= 65) & (
        df["Gender"] == "Female"), "NEW_GENDER_CAT"] = "YouthFemale"
    # ORTA YAŞ ERKEK-KADIN 66-79
    df.loc[(df["Age"] >= 66) & (df["Age"] <= 79) & (
        df["Gender"] == "Male"), "NEW_GENDER_CAT"] = "MiddleAgeMale"
    df.loc[(df["Age"] >= 66) & (df["Age"] <= 79) & (
        df["Gender"] == "Female"), "NEW_GENDER_CAT"] = "MiddleAgeFemale"
    # YAŞLI
    df.loc[(df["Age"] > 79) & (df["Gender"] == "Female"),
           "NEW_GENDER_CAT"] = "OldAgeFemale"
    df.loc[(df["Age"] > 79) & (df["Gender"] == "Male"),
           "NEW_GENDER_CAT"] = "OldAgeMale"
    return df


def asel_mebaysan_preprocess(na='drop'):
    df = get_data()
    if na == 'drop':
        df = df.dropna()
    else:
        # df.groupby(['Type of Travel','satisfaction']).agg({'Departure Delay in Minutes':'mean'}) # ToDO: silmeden eksik değerleri grup kırılımında impute edebiliriz
        pass
    df = age_engineering(df)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


df = asel_mebaysan_preprocess()

df.head()


