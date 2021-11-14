from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from xgboost.training import cv


def random_forest_classifier(X, y):
    rf_model = RandomForestClassifier(
        random_state=34).fit(X, y)  # model nesnesi oluşturuyoruz

    # hyperparameter optimizasyonu yapmak için hiperparametreleri yazıyorum
    # rf_params = {"max_depth": [5, 8, None],  # max ağaç derinliği
    #              # bölünme işlemi yapılırken göz önünde bulundurulacak olan değişken sayısı
    #              "max_features": [3, 5, 7, "auto"],
    #              # bir node'u dala ayırmak için gerekli minimum gözlem sayısı
    #              "min_samples_split": [2, 5, 8, 15, 20],
    #              "n_estimators": [100, 200, 500]}  # ağaç sayısı, kolektif (topluluk) öğrenme metodu olduğundan kaç adet ağaç olmasını istiyoruz

    # # en iyi parametreleri arıyoruz
    # rf_best_grid = GridSearchCV(rf_model,  # hangi model
    #                             rf_params,  # hangi parametreler
    #                             cv=3  # kaç katlı çaprazlama
    #                             ).fit(X, y)

    # # final model kuruyoruz
    # rf_final = rf_model.set_params(**rf_best_grid.best_params_,  # en iyi hiperparametreleri modele set ediyorum
    #                                random_state=34).fit(X, y)

    # final model cv ile test ediyoruz
    cv_results = cross_validate(rf_model,  # final modelin cross validation hatası
                                X, y,
                                cv=3,
                                scoring=["accuracy", "f1", "roc_auc"])

    return {
        'model': rf_model,
        'train_accuracy': cv_results['test_accuracy'].mean(),
        'train_f1': cv_results['test_f1'].mean(),
        'train_roc_auc': cv_results['test_roc_auc'].mean()
    }


def xgbm_classifier(X, y):
    xgboost_model = XGBClassifier(random_state=34).fit(X, y)

    # xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],  # büyüme şiddeti
    #                   "max_depth": [5, 8, 12, 15, 20],
    #                   # ağaç sayısı, iterasyon sayısı..
    #                   "n_estimators": [100, 500, 1000],
    #                   # yüzdelik olarak kaç gözlem bulunsun
    #                   "colsample_bytree": [0.5, 0.7, 1]
    #                   }

    # xgboost_best_grid = GridSearchCV(xgboost_model,
    #                                  xgboost_params,
    #                                  cv=5,
    #                                  verbose=True).fit(X, y)

    # xgboost_best_grid.best_score_

    # xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_,
    #                                          random_state=17).fit(X, y)

    cv_results = cross_validate(xgboost_model,
                                X, y,
                                cv=10,
                                scoring=["accuracy", "f1", "roc_auc"])
    return {
        'model': xgboost_model,
        'train_accuracy': cv_results['test_accuracy'].mean(),
        'train_f1': cv_results['test_f1'].mean(),
        'train_roc_auc': cv_results['test_roc_auc'].mean()
    }
