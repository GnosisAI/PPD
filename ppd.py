from sklearn.metrics import pairwise_distances
from random import randint
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import ( normalized_mutual_info_score as nmi , 
                             accuracy_score as acc , 
                             adjusted_rand_score as ari )

class SDRC(BaseEstimator):
    def __init__(self, K, l=0.001):
        self.K = K
        self.l = l
        self.hot_encoder = OneHotEncoder(sparse=False, categories='auto')

    def fit(self, X, epochs=10):
        self.nrows = X.shape[0]
        # initalisation
        B = np.random.rand(self.nrows, self.K)
        Z = np.array([randint(1, self.K) for _ in range(self.nrows)])
        self.hot_encoder.fit(Z.reshape(-1, 1))
        Z = self.hot_encoder.transform(Z.reshape(-1, 1))
        S = pairwise_distances(X)
        S /= S.max()

        for epoch in tqdm(range(epochs)):
            Z, Q, B, z = self.train_step(S, Z, B)
        return Z, B, z

    def train_step(self, S, Z, B):
        Uq, s, Vq = np.linalg.svd(Z.T.dot(B))
        Q = Uq.dot(Vq.T)

        M = S.T.dot(B)
        tmp = (S.T.dot(M) + self.l * Z.dot(Q))

        Ub, s, Vb = np.linalg.svd(tmp, full_matrices=False)
        B = Ub.dot(Vb.T)

        z = np.array(
            [np.argmax(np.linalg.norm(B[i] - Q, axis=1)) + 1 for i in range(self.nrows)])
        Z = self.hot_encoder.transform(z.reshape(-1, 1))
        return Z, Q, B, z
    
    def plot_results(self, B, hue, axis = (0, 1), **kwargs):
        if np.issubdtype(hue.dtype, np.number):
            hue = [str(x) + '_' for x in hue]
        plt.subplots(figsize=(10, 10))
        ax1, ax2 = axis
        return sns.scatterplot(B[:, ax1], B[:, ax2], hue=hue, **kwargs)
    
    def clustering_report(self, y_true, y_pred):
        return {
            "accuracy": acc(y_true, y_pred),
            "nmi": nmi(y_true, y_pred, average_method='arithmetic'),
            "ari": ari(y_true, y_pred)
        }