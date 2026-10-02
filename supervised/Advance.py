from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

X,y = load_iris(return_X_y=True)
X_train , X_test , y_train , y_test = train_test_split (X,y,random_state=42)

# bagging reduce variance
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print("Random Forest Accuracy:",accuracy_score(y_test,y_pred))


# boosting increase accuracy
xg = xgb.XGBClassifier(n_estimators=100,learning_rate=0.1,random_state=42)
xg.fit(X_train,y_train)
y_xg_pred = xg.predict(X_test)
print("Xgboost Accuracy:",accuracy_score(y_test,y_xg_pred))


# stacking Combine different models to get a stronger one.
# provide excellent result in churn project than individual result
base_learner = [
    ("svc",SVC(probability=True)),
    ("dt",DecisionTreeClassifier())
]
meta_learner = LogisticRegression()
stack_class = StackingClassifier(estimators=base_learner,final_estimator=meta_learner)
stack_class.fit(X_train ,y_train)
y_stack_pred = stack_class.predict(X_test)
print("stack Accuracy:",accuracy_score(y_stack_pred ,y_test))

# LightGBM
import lightgbm as lgb
from sklearn.metrics import accuracy_score

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss'
}

lgb_model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=50, early_stopping_rounds=10)
y_pred = lgb_model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_classes))

# CatBoost
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0)
cat_model.fit(X_train, y_train)
y_pred = cat_model.predict(X_test)
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred))

# their are 2 more type of regression :-

# 1. Bayesian regression :- instead of giving single output , it provide range of output ,
# example :- house prediction model with normal regression provide single output and this will provide a range price (100 - 120)

# 2. Quantile regression = percentile-focused predictions.
# other regression provide average value while this can use to spread of possible outcome for example :-
# master degree salary prediction, normal regression output 80000,
# with Quantile regression you can get what income does top 10% earn or 90%

