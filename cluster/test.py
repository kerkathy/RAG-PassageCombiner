# %%
import json
from time import time

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

# %%
def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)

data_file = "../ic-ralm-odqa/in-context-ralm/reproduce_retrieval/result/nq-test/formatted-ms2.nq-test.hits-100-reranked-mmr-0.9-tfidf-1-debug.json"
data = load_dataset(data_file)

# print(type(data))
# print(len(data))
# print(data[0].keys())
# print(data[0]["question"])
# print(data[0]["answers"])
# print(data[0]["ctxs"][0].keys())
# print(data[0]["ctxs"][0])

def get_contexts(qid):
    """ Collect all documents for question i """
    return [x["text"] for x in data[qid]["ctxs"]]

# %%
ctx_0 = get_contexts(0)
for i, ctx in enumerate(ctx_0):
    print(f"{i}. {ctx}")
    if i == 10:
        break

def show_wordcloud(text):
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Get the text of the first document
show_wordcloud(ctx_0[0])
show_wordcloud(ctx_0[1])
show_wordcloud(ctx_0[2])

# show word cloud for all 100 documents
show_wordcloud(" ".join(ctx_0))

# %%
vectorizer = TfidfVectorizer()
X_doc_tfidf = vectorizer.fit_transform(ctx_0)

question = data[0]["question"]
X_question_tfidf = vectorizer.transform([question])

# %%
vocab = vectorizer.vocabulary_
# sort the vocab by value
vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))

# %%
evaluations = []
evaluations_std = []
k = 3

def plot_only_2d(X, y, name):
    plt.figure(figsize=(5, 5))
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=label)
    plt.title(name)
    plt.legend()
    plt.show()

def train(km, X, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)

    return km

# %%
def dim_reduction(X, n_components=2):
    lsa = make_pipeline(TruncatedSVD(n_components=n_components), Normalizer(copy=False))
    return lsa.fit_transform(X)

# %%
n_components = 100
n_clusters = 3

# Create a pipeline that first applies TruncatedSVD and then Normalizer
lsa = make_pipeline(TruncatedSVD(n_components=n_components), Normalizer(copy=False))

# Fit and transform the documents using the pipeline
lsa.fit(X_doc_tfidf)
X_doc_lsa = lsa.transform(X_doc_tfidf)

# Transform the questions using the fitted pipeline
X_question_lsa = lsa.transform(X_question_tfidf)

kmeans = KMeans(
    n_clusters=n_clusters,
    max_iter=100,
    n_init=1,
)
trained_km = train(kmeans, X_doc_lsa)

# %%
import matplotlib.cm as cm

def plot_special(X_doc, X_question, y, name, type):
    if type == "PCA":
        pca = PCA(n_components=2).fit(X_doc)
        X_embedded = pca.transform(X_doc)
        question_embedded = pca.transform(X_question)
    elif type == "TSNE":
        tsne = TSNE(n_components=2).fit(X_doc)
        X_embedded = tsne.fit_transform(X_doc)
        question_embedded = tsne.transform(X_question)
    else:
        raise ValueError(f"Unknown type: {type}")
    plt.figure(figsize=(5, 5))

    # Create a colormap
    cmap = cm.get_cmap('Accent')
    # cmap = cm.get_cmap('Set2')

    # Calculate the number of unique labels and create a color for each one
    labels = np.unique(y)
    colors = [cmap(i) for i in range(len(labels))]

    for label, color in zip(labels, colors):
        plt.scatter(X_embedded[y == label, 0], X_embedded[y == label, 1], label=label, color=color)
    plt.scatter(question_embedded[:, 0], question_embedded[:, 1], label='question', color='r')

    plt.title(name)
    plt.legend()
    plt.show()
# predict X using the trained model
y_pred = trained_km.predict(X_doc_lsa)

# %%
plot_special(X_doc_lsa, X_question_lsa, y_pred, "KMeans", "PCA")
# plot_only_2d(X, y_pred, f"{name} (first 2 dim)")

# %%
# get the cluster centroid and sort them by distance
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(k):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :10]:
        print(f"{terms[ind]} ", end="")
    print()

# print the cluster
print(y_pred)
print(len(y_pred))

# %%
# For each centroid, sort all documents in the cluster by distance to centroid

def get_cluster_docs(X, y_pred, kmeans, cluster_id, n_docs=10):
    # get the distance to the centroid
    dist = kmeans.transform(X)[:, cluster_id]
    # get the indices of the documents in the cluster
    cluster_indices = np.where(y_pred == cluster_id)[0]
    # sort the indices by distance to the centroid
    sorted_indices = cluster_indices[np.argsort(dist[cluster_indices])]
    # return the top n_docs
    return sorted_indices[:n_docs]

for i in range(k):
    print(f"Cluster {i}:")
    for idx in get_cluster_docs(X_doc_lsa, y_pred, kmeans, i):
        print(f"{idx}: {ctx_0[idx]}")
    print()
# %%
