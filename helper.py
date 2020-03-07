import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

s = StandardScaler()
l = LabelEncoder()

def load_dataset(dataset, base_path='./clean_data', max_sample=None):
    path = f"{base_path}/{dataset}"
    X = np.load(path+'_img.npy')
    old_shape = X.shape
    X = X.reshape(X.shape[0], -1)
    X = s.fit_transform(X)
    
    y = np.load(path+'_lbl.npy')
    y = l.fit_transform(y)
    
    if max_sample:
        X = X[:max_sample]
        y = y[:max_sample]
    
    return (X, y, old_shape)

def print_results(data, shape=(28,28), n=10, encoded_imgs=None):
    
    pdata = data.reshape(shape)
    plt.figure(figsize=(40, 10))
    for i in range(n):        
        ax = plt.subplot(4, 20, i + 1)
        plt.imshow(pdata[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if encoded_imgs is not None:
            ax = plt.subplot(4, 20, 2*20+i + 1 )
            plt.imshow(encoded_imgs[i].reshape(-1, 2))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
