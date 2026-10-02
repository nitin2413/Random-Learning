# the data on which we were working was very much imbalance

# To avoid the imbalance problem:
#•	Use class_weight="balanced" in models,

# in imbalance data we should not measure accuracy we should measure with Recall


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/nitinsingh/PycharmProjects/ML_Project/project/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#if null value exist clearing them
df = df.dropna()
df = df.drop('customerID',axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])  # drop rows with NaN in TotalCharges

# re balancing the data (chatgpt)
churn_yes = df[df["Churn"] == "Yes"]
churn_no = df[df["Churn"] == "No"]
churn_yes_upsampled = resample(churn_yes, replace=True, n_samples=len(churn_no), random_state=42)
df_balanced = pd.concat([churn_no, churn_yes_upsampled])
# what is happening in this region of code
# resample(churn_yes, replace=True, n_samples=len(churn_no))
# 	•	Takes the minority class (churn_yes).
# 	•	Randomly duplicates rows (with replacement).
# 	•	Expands it until it has the same number of rows as the majority class.
##################################

df = df_balanced.copy()

numerical_data = df.select_dtypes(include=['number'])
print(numerical_data)

categorical_data = df.select_dtypes(include=['object','category'])
print(categorical_data)

#endocing section
encoder = OneHotEncoder(drop= None)
encoded_data= encoder.fit_transform(categorical_data)

encoded_df = pd.DataFrame(
    encoded_data.toarray(),
    columns=encoder.get_feature_names_out(categorical_data.columns)
)

print(encoded_df.head())

# scaling numerical data
minmax = MinMaxScaler()
scaled_data = minmax.fit_transform(numerical_data)

print(scaled_data)

# merge scales_data and encoded_data
# Final dataset = scaled numerical + encoded categorical
final_data = pd.concat(
    [pd.DataFrame(scaled_data, columns=numerical_data.columns).reset_index(drop=True),
     encoded_df.reset_index(drop=True)],
    axis=1
)

print(final_data.head())

# split data
from sklearn.utils import resample

#spliting data
X = final_data.drop(columns= ["Churn_Yes","Churn_No"])
y = final_data["Churn_Yes"]

X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Select only numerical features
num_features = df.select_dtypes(include=[np.number])

# Correlation matrix
corr_matrix = num_features.corr()

# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Features")
plt.show()

contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
print(contract_churn)

contract_churn.plot(kind="bar", stacked=True, figsize=(6,4))
plt.title("Churn % by Contract Type")
plt.ylabel("Percentage")
plt.show()


# from logistic_model import LogisticRegression
# logi = LogisticRegression(lr=0.01, n_iters=1000)
# # logi.fit(X_train.to_numpy(), y_train.to_numpy())# as my logistic_model is writen using numpy only they required numpy array instead of pandas dataframe
# # logi.predict(X_test.to_numpy())
#
# logi.fit(X_train.values, y_train.values)  # convert to numpy arrays
# y_pred = logi.predict(X_test.values)
# print("First 10 predictions:", y_pred[:10])

from sklearn.linear_model import LogisticRegression

log = LogisticRegression(max_iter=1000)
log.fit(X_train ,y_train )
y_pred = log.predict(X_test)
print(y_pred )
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("logistic:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train ,y_train)
y_pre = tree.predict(X_test)
print("decision tree:")
print("Accuracy:", accuracy_score(y_test, y_pre))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pre))
print("\nClassification Report:\n", classification_report(y_test, y_pre))

forest = RandomForestClassifier(max_depth=5)#class_weight="balanced"This makes the model pay more attention to churners without duplicating data.
forest.fit(X_train ,y_train)
y_pr = forest.predict(X_test)
print("random forest:")
print("Accuracy:", accuracy_score(y_test, y_pr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pr))
print("\nClassification Report:\n", classification_report(y_test, y_pr))

xg = XGBClassifier(max_depth=5)
xg.fit(X_train ,y_train)
y_p = xg.predict(X_test)
print("XGBOOST:")
print("Accuracy:", accuracy_score(y_test, y_p))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_p))
print("\nClassification Report:\n", classification_report(y_test, y_p))

base_lerner = [
    ("xg",XGBClassifier()),
    ("rdf",RandomForestClassifier()),
    ("dt",DecisionTreeClassifier())
]
meta_learner = RandomForestClassifier()

stack_class = StackingClassifier(estimators=base_lerner,final_estimator=meta_learner)
stack_class.fit(X_train ,y_train )
y_stack_pred = stack_class.predict(X_test)
print("accuracy:",accuracy_score(y_test,y_stack_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_stack_pred))
print("\nClassification Report:\n", classification_report(y_test, y_stack_pred))



