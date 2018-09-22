import numpy as np
import pandas as pd
df = pd.read_csv("data.csv", sep=";")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, random_state=17)

def confuse(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("\nConfusion Matrix: \n", cm)
    fpr(cm)
    ffr(cm)

def fpr(confusion_matrix):
    fp = confusion_matrix[0][1]
    tf = confusion_matrix[0][0]
    rate = float(fp) / (fp + tf)
    print("False Pass Rate: ", rate)


def ffr(confusion_matrix):
    ff = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]
    rate = float(ff) / (ff + tp)
    print("False Fail Rate: ", rate)

    return rate
def train_and_score(X, y):
    print("train and score")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    clf = Pipeline([
        ('reduce_dim', SelectKBest(chi2, k=2)),
        ('train', LinearSVC(C=100))
    ])
    
    scores = cross_val_score(clf, X_train, y_train)
    
    print("Mean Model Accuracy:", np.array(scores).mean())

    clf.fit(X_train, y_train)
    data=pd.read_csv("datatest.csv",sep=";")
    class_le = LabelEncoder()
    for column in data[["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]].columns:
        data[column] = class_le.fit_transform(data[column].values)
    if data["G1"][0]>10:
        data["G1"][0]=1
    else:
        data["G1"][0]=0
        
    if data["G2"][0]>10:
        data["G2"][0]=1
    else:
        data["G2"][0]=0
    trp=clf.predict(data)
    print("Predicted Data ",trp)
    confuse(y_test, clf.predict(X_test))
    print()
    
    
def main():
    print("\nStudent Performance Prediction")

    class_le = LabelEncoder()
    for column in df[["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]].columns:
        df[column] = class_le.fit_transform(df[column].values)

    
    for i, row in df.iterrows():
        if row["G1"] >= 10:
            df["G1"][i] = 1
        else:
            df["G1"][i] = 0

        if row["G2"] >= 10:
            df["G2"][i] = 1
        else:
            df["G2"][i] = 0

        if row["G3"] >= 10:
            df["G3"][i] = 1
        else:
            df["G3"][i] = 0
    
    y = df.pop("G3")
    X = df  
    print("\n\nModel Accuracy Knowing G1 & G2 Scores")
    train_and_score(X, y)
    
   

main()
