from sklearn.cluster import KMeans

def run_kmeans(X, k=3):
    model = KMeans(n_clusters=k)
    return model.fit_predict(X)
