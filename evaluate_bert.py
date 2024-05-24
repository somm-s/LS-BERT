from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import tqdm
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import joblib


def get_embeddings(model_path, data, batch_size=256, num_samples=5000, resample=True, use_labels=True, model=None):
    """
    Get embeddings from a BERT model. Take random samples from the data.
    """
    # Randomly sample data
    if len(data) > num_samples and resample:
        data = np.random.choice(data, num_samples, replace=False)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    if model is None:
        model = AutoModel.from_pretrained(model_path)
        model.to('cuda')
        model.eval()

    embeddings = []
    
    if use_labels:
        labels = []
    
    for batch in tqdm.tqdm(data_loader):
        if use_labels:
            labels.extend(batch['label'].numpy())
        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            outputs = model.base_model(**batch)
        embeddings.extend(outputs.last_hidden_state[:,0,:].cpu().numpy())
        
    if use_labels:
        return np.array(embeddings), np.array(labels)
    return np.array(embeddings)


def infer_label(fine_tuned_model_path, data, batch_size, softmax_threshold=0.5, model=None, use_labels=False):
    """
    Infer labels from a fine-tuned BERT model.
    """
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
        model.to('cuda')
        model.eval()

    if use_labels:
        true_labels = []


    labels = []
    
    probas = []
    for batch in tqdm.tqdm(data_loader):
        if use_labels:
            true_labels.extend(batch['label'].numpy())

        batch = {k: v.to('cuda') for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        labels.extend((probs[:,1] >= softmax_threshold).cpu().numpy())
        probas.extend(probs[:,1].cpu().numpy())
    
    if use_labels:
        return np.array(labels), np.array(probas), np.array(true_labels)
    return np.array(labels), np.array(probas)


def visualize_embeddings(embeddings_list, legend_list, title, method='pca', sample=None, x_min=None, x_max=None, y_min=None, y_max=None, alpha=0.5):
    """
    Visualize embeddings using PCA or t-SNE. embeddings_list is a list of numpy arrays
    containing embeddings from different datasets (train, test, validation, etc.)
    """
    
    if sample is not None:
        embeddings_list = [x[np.random.choice(len(x), sample, replace=False)] for x in embeddings_list]

    embeddings = StandardScaler().fit_transform(np.concatenate(embeddings_list))
    labels = np.concatenate([np.full(len(x), i) for i, x in enumerate(embeddings_list)])

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2)
    elif method == 'kpca':
        reducer = KernelPCA(n_components=2, kernel='rbf')
    else:
        raise ValueError("Invalid method. Expected 'pca' or 'tsne'")

    principalComponents = reducer.fit_transform(embeddings)

    # save trained pca to disk using joblib
    joblib.dump(reducer, "pca.pkl")
    
    plt.figure(figsize=(10,10))
    for i, emb in enumerate(embeddings_list):
        indices = labels == i
        plt.scatter(principalComponents[indices, 0], principalComponents[indices, 1], alpha=alpha, label=legend_list[i])
    
    if x_min and x_max and y_min and y_max:
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
    
    plt.legend()
    plt.title(title)
    plt.show()
    
    
def evaluate_classifier(X_0, X_1, X_0_test, X_1_test, classifier, threshold=0.5, hyperparameters={}):
    """
    Evaluate a classifier on train and test data. 
    """
    if classifier == "SVM":
        from sklearn.svm import SVC
        clf = SVC(**hyperparameters, probability=True)
    elif classifier == "RF":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**hyperparameters)
    elif classifier == "LR":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(**hyperparameters)
    else:
        raise ValueError("Invalid classifier. Expected 'SVM', 'RF', or 'LR'")
        
    X_train = np.concatenate([X_0, X_1])
    y_train = np.concatenate([np.zeros(len(X_0)), np.ones(len(X_1))])
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    X_test = np.concatenate([X_0_test, X_1_test])
    y_test = np.concatenate([np.zeros(len(X_0_test)), np.ones(len(X_1_test))])
    
    clf.fit(X_train, y_train)
    
    y_proba = clf.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= threshold).astype(int)
    
    print(classification_report(y_test, y_pred))
    
    fpr = np.sum((y_pred == 1) & (y_test == 0)) / np.sum(y_test == 0)
    print("False Positive Rate:", fpr)
    
    return clf