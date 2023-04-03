import pandas as pd

train_data, test_data = pd.read_csv("titanic/train.csv"), pd.read_csv("titanic/test.csv")
train_data.head()

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")
train_data.info()

train_data[train_data["Sex"]=="female"]["Age"].median()

train_data.describe()

train_data["Survived"].value_counts()

train_data["Pclass"].value_counts()

train_data["Sex"].value_counts()

train_data["Embarked"].value_counts() # C = Cherbourg, Q = Queenstown, S = Southampton 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),    
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

X_train = preprocess_pipeline.fit_transform(train_data)

y_train = train_data["Survived"]

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)

print(y_pred)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

print(forest_scores.mean())

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)

print(svm_scores.mean())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()
















#train_data["AgeBucket"] = train_data["Age"] // 15 * 15
#train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
#
#train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
#train_data[["RelativesOnboard", "Survived"]].groupby(
#    ['RelativesOnboard']).mean()