# Cough Detection using an SVM

The `Net` class in `net.py` creates the network architecture detailed in [this](https://arxiv.org/abs/1711.01369) paper and loads the parameters from `mx-h64-1024_0d3-1.17.pkl`. This network is then used in `extract_features.py` to extract 1024-dimensional feature vectors from files in this [dataset](), which are saved in `X.npy` and their associated labels are saved in `labels.pkl`. This implementation is loosely based on [this](https://github.com/anuragkr90/weak_feature_extractor) one. Finally, the features vectors in `X.npy` and their labels in `labels.pkl` are used in `main.py` to train a support vector machine with an RBF kernel to classify coughs and non-coughs.

