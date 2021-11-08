from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset_depparse import *
import numpy as np


base_dir = 'datasets/rest/'
data = Dataset(base_dir, is_preprocessed=True)
X_train_raw = [" ".join(x.words) for x in data.train_data if x.aspect_cluster == 0]
X_test_raw = [" ".join(x.words) for x in data.test_data if x.aspect_cluster == 0]
y_train_labels = np.asarray([x.polarity for x in data.train_data if x.aspect_cluster == 0])
y_test_labels = np.asarray([x.polarity for x in data.test_data if x.aspect_cluster == 0])

print(len(X_train_raw))
print(len(X_test_raw))
print(len(y_train_labels))
print(len(y_test_labels))

vectorizer = TfidfVectorizer(token_pattern=r'\w{1,}')
X_train_tfidf = vectorizer.fit_transform(X_train_raw).toarray()
print(X_train_tfidf.shape)

svd = TruncatedSVD(10, algorithm='arpack', random_state=42, n_iter=5000)
lsa = make_pipeline(svd, Normalizer(copy=False))
X_train_lsa = lsa.fit_transform(X_train_tfidf)
print(X_train_lsa.shape)

X_test_tfidf = vectorizer.transform(X_test_raw).toarray()
X_test_lsa = lsa.transform(X_test_tfidf)

knn1 = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn1.fit(X_train_tfidf, y_train_labels)
p1 = knn1.predict(X_test_tfidf)
print(len(p1))

knn2 = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn2.fit(X_train_lsa, y_train_labels)
p2 = knn2.predict(X_test_lsa)
print(len(p2))


acc1 = accuracy_score(y_test_labels, p1)
acc2 = accuracy_score(y_test_labels, p2)
print(acc1)
print(acc2)




