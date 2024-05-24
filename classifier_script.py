from similarity_dataset import SimilarityDataset
data_2021 = SimilarityDataset(2021, fine=True, label_path="ip_labels_2021.txt", only_rt=False)
data_2022 = SimilarityDataset(2022, fine=True, label_path="ip_labels_2022.txt", only_rt=False)

import numpy as np

X_rt_21, X_rt_ip_21 = data_2021.get_rt_host_embeddings()
X_b_21, X_b_ip_21 = data_2021.get_benign_host_embeddings()
X_rt_22, X_rt_ip_22 = data_2022.get_rt_host_embeddings()
X_b_22, X_b_ip_22 = data_2022.get_benign_host_embeddings()
XX_rt_21, XX_rt_ip_21 = data_2021.get_rt_host_host_embeddings()
XX_b_21, XX_b_ip_21 = data_2021.get_benign_host_host_embeddings()
XX_rt_22, XX_rt_ip_22 = data_2022.get_rt_host_host_embeddings()
XX_b_22, XX_b_ip_22 = data_2022.get_benign_host_host_embeddings()

X_21 = np.concatenate([X_rt_21, X_b_21])
y_21 = np.concatenate([np.ones(X_rt_21.shape[0]), np.zeros(X_b_21.shape[0])])
X_22 = np.concatenate([X_rt_22, X_b_22])
y_22 = np.concatenate([np.ones(X_rt_22.shape[0]), np.zeros(X_b_22.shape[0])])
XX_21 = np.concatenate([XX_rt_21, XX_b_21])
yy_21 = np.concatenate([np.ones(XX_rt_21.shape[0]), np.zeros(XX_b_21.shape[0])])
XX_22 = np.concatenate([XX_rt_22, XX_b_22])
yy_22 = np.concatenate([np.ones(XX_rt_22.shape[0]), np.zeros(XX_b_22.shape[0])])

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_model(X, y, model, param_grid, cv=5):
    skf = StratifiedKFold(n_splits=cv)
    clf = GridSearchCV(model, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
    clf.fit(X, y)
    return clf

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

model = SVC(probability=True)
clf = train_model(X_22, y_22, model, param_grid)

# validate on 2021 data
y_pred = clf.predict_proba(X_21)[:,1]
print(roc_auc_score(y_21, y_pred))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

model = LogisticRegression()
clf = train_model(X_22, y_22, model, param_grid)

# validate on 2021 data
y_pred = clf.predict_proba(X_21)[:,1]
print(roc_auc_score(y_21, y_pred))

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 20, 30]
}

model = RandomForestClassifier()
clf = train_model(X_22, y_22, model, param_grid)

# validate on 2021 data
y_pred = clf.predict_proba(X_21)[:,1]
print(roc_auc_score(y_21, y_pred))

# repeat everything for host-host embeddings
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

model = SVC(probability=True)
clf = train_model(XX_22, yy_22, model, param_grid)

# validate on 2021 data
y_pred = clf.predict_proba(XX_21)[:,1]
print(roc_auc_score(yy_21, y_pred))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

model = LogisticRegression()
clf = train_model(XX_22, yy_22, model, param_grid)

# validate on 2021 data
y_pred = clf.predict_proba(XX_21)[:,1]
print(roc_auc_score(yy_21, y_pred))

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 20, 30]
}

model = RandomForestClassifier()
clf = train_model(XX_22, yy_22, model, param_grid)

# validate on 2021 data
y_pred = clf.predict_proba(XX_21)[:,1]

print(roc_auc_score(yy_21, y_pred))