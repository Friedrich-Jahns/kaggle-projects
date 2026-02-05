import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dat_train = pd.read_csv('dat/train.csv')
dat_test = pd.read_csv('dat/test.csv')

class data:
    def __init__(self,dat,mode='train',clf=None):
        self.raw = dat
        self.clf = clf
        self.id = dat['id']
        if mode == 'train':
            self.X = dat.drop(columns=['Heart Disease'])
            self.y = dat['Heart Disease']
            self.le = LabelEncoder()
            self.y_encoded = self.le.fit_transform(dat["Heart Disease"])
        else:
            self.X = dat

        self.onehot = pd.get_dummies(self.X)
        self.std_scaler = StandardScaler()
        self.onehot_scaled = self.std_scaler.fit_transform(self.onehot)
        
    def train_test(self):
        X_train,X_val,y_train,y_val = train_test_split(
            self.onehot_scaled, self.y_encoded,
            test_size=0.2,
            random_state=42
        )
        return X_train,X_val,y_train,y_val
    def corr_matrix(self):
        df = self.X.copy()
        df["Heart Disease"] = self.y_encoded
        corr = df.corr(numeric_only=True)
        corr_cpy = corr.copy()
        corr_cpy = np.where(corr_cpy==1,0,corr)
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_cpy)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Korrelationsmatrix")
        plt.tight_layout()
        plt.show()

    def RandomForrestClassifier(self,n_estimators=100, test_size=0.2, random_state=42):
        X_train,X_val,y_train,y_val = self.train_test()
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")
        
        return clf
   
    def RFC_RandomSearch(self,n_iter=20, random_state=42):
        X_train,X_val,y_train,y_val = self.train_test()
        clf = RandomForestClassifier(random_state=random_state)
        
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }
        rand_search = RandomizedSearchCV(
            clf,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            scoring="accuracy",
            random_state=random_state,
            n_jobs=-1,
            verbose=2
        )
        rand_search.fit(X_train, y_train)

        best_model = rand_search.best_estimator_
        y_pred = best_model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        print(f"Best params: {rand_search.best_params_}")
        print(f"Validation Accuracy: {acc:.4f}")
        return best_model

    def RandomForrestClassifier_pred(self):
        model = self.clf
        y_pred = model.predict(self.X)
        submission = pd.DataFrame({
            "id": self.id,
            "Heart Disease": y_pred
        })
        submission.to_csv("submission.csv", index=False)
        print('saved')

train = data(dat_train)
print(dat_train.head())
#clf = train.RandomForrestClassifier()
clf = train.RFC_RandomSearch()
test = data(dat_test,mode='pred',clf=clf)
test.RandomForrestClassifier_pred()
