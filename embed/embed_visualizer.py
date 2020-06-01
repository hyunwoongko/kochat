"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc, rcParams
from sklearn.manifold import TSNE


class EmbedVisualizer:

    def __init__(self, config):
        self.conf = config
        font_path = 'C:/Windows/Fonts/batang.ttc'
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
        rcParams.update({'figure.autolayout': True})

    def plot_with_labels(self, low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), \
            "More labels than embeddings"

        plt.figure(figsize=(22, 22))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(10, 4),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)
        plt.show()

    def visualize(self, model):
        words = []
        embedding = np.array([])

        for i, word in enumerate(model.wv.vocab):
            words.append(word)
            embedding = np.append(embedding, model[word])

        embedding = embedding.reshape(len(words), self.conf.vector_size)
        tsne = TSNE(perplexity=10.0, n_components=2, init='pca', n_iter=15000)
        low_dim_embedding = tsne.fit_transform(embedding)
        self.plot_with_labels(low_dim_embedding, words)
