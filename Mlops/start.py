from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib
from torch.distributed.pipelining import pipeline

pipeline = Pipeline([
    ("scaler" , StandardScaler()),
    ("model" , LogisticRegression())
])

X_train , y_train = load_iris(return_X_y=True)

pipeline.fit(X_train , y_train)

joblib.dump(pipeline , "model.pkl")

model = joblib.load("model.pkl")
predict = model.predict(X_train[:5])
print(predict)

