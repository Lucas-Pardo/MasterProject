
import numpy as np
import pandas as pd
from sklearn import svm, metrics, neighbors, multioutput, multiclass, linear_model, pipeline, \
                    discriminant_analysis, kernel_ridge, gaussian_process, cross_decomposition, tree, ensemble, \
                    model_selection, calibration, neural_network
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Parameters
usedModel = "gpc"   # Model: {knn, svc, gpc, vc}

# Import data

data = pd.read_excel("Data/prediction_data1.xlsx")

types = {}

for x in data["P.Type"].values:
    if x in types:
        types[x] += 1
    else:
        types[x] = 1


# print(data.head())
# print(types)

# Type classification

def split(pwd_types, features):
    feat_names = ["P.Conc", "Voltage", "Mt frac", "Radius avg", "Depth avg", "Sa avg"]
    train = pd.DataFrame(columns=features.columns)
    test = pd.DataFrame(columns=features.columns)

    for c in pwd_types.keys():
        aux = features.loc[features["P.Type"] == c]
        tr, ts = train_test_split(aux, test_size=0.15)
        train = pd.concat([train if not train.empty else None, tr], ignore_index=True)
        test = pd.concat([test if not test.empty else None, ts], ignore_index=True)

    xtrain = train[feat_names].values
    ytrain = train["P.Type"].values
    xtest = test[feat_names].values
    ytest = test["P.Type"].values

    return xtrain, ytrain, xtest, ytest


# xtrain, ytrain, xtest, ytest = split(types, data)
# print(len(ytrain), len(ytest), len(ytrain)+len(ytest))

# Convert powder types into numbers

def labelT(ytrain, ytest):
    le = LabelEncoder()
    ytrain = le.fit_transform(np.ravel(ytrain))
    ytest = le.transform(np.ravel(ytest))
    return ytrain, ytest


def gen_c(n, start, finish):
    p = (finish - start)/n
    y = start
    for i in range(n):
        yield y
        y += p


# Models

if usedModel.lower() == "knn":
    ### KNN:
    best = [0, 0, 0, 0]
    weight = ["uniform", "distance"]
    metrics = ["euclidean", "manhattan", "chebyshev", "canberra", "braycurtis"]
    history = {}

    for metric in metrics:
        hist = np.zeros((2, 20))
        for w in range(len(weight)):
            for n in range(1, 21):
                bestrandom = 0
                for i in range(10):
                    xtrain, ytrain, xtest, ytest = split(types, data)
                    ytrain, ytest = labelT(ytrain, ytest)
                    base = neighbors.KNeighborsClassifier(n_neighbors=n, weights=weight[w], metric=metric)
                    model = pipeline.make_pipeline(StandardScaler(), base)
                    model.fit(xtrain, ytrain)
                    acc = model.score(xtest, ytest)
                    if acc > bestrandom:
                        bestrandom = acc
                hist[w, n-1] = bestrandom
                if bestrandom > best[3]:
                    best = [metric, weight[w], n, bestrandom]
        history[metric] = hist


    print("Best: (Metric: {}, Weights: {}, Neighbors: {}, acc: {:.2%})".format(best[0], best[1], best[2], best[3]))


    x_pos = np.arange(start=1, stop=21)
    relPos = [-0.2, -0.1, 0, 0.1, 0.2]
    colors = ["orangered", "gold", "limegreen", "dodgerblue", "violet"]
    for w in range(len(weight)):
        fig, ax = plt.subplots()
        ax.set_xticks(x_pos, labels=x_pos, rotation=0)
        ax.set_xlabel("Neighbors")
        ax.set_ylabel("Accuracy")
        ax.set_title("Weights: " + weight[w])
        for i in range(5):
            ax.bar(x_pos+relPos[i], history[metrics[i]][w, :], width=0.1, color=colors[i])
        ax.legend(labels=metrics)
        plt.show()

elif usedModel.lower() == "svc":
    ### SVC
    best = [0, 0, 0]
    kernels = ["linear", "poly", "rbf"]#, "sigmoid"]
    history = {}
    nc = 50

    for kernel in kernels:
        hist = np.zeros((nc,))
        k = 0
        for c in gen_c(nc, 1, 20):
            bestrandom = 0
            for i in range(5):
                xtrain, ytrain, xtest, ytest = split(types, data)
                ytrain, ytest = labelT(ytrain, ytest)
                base = svm.SVC(kernel=kernel, C=c)
                model = pipeline.make_pipeline(StandardScaler(), base)
                model.fit(xtrain, ytrain)
                acc = model.score(xtest, ytest)
                if acc > bestrandom:
                    bestrandom = acc
            hist[k] = bestrandom
            if bestrandom > best[2]:
                best = [kernel, c, bestrandom]
            k += 1
            print(kernel, k/nc)
        history[kernel] = hist


    print("Best: (Kernel: {}, C: {}, acc: {:.2%})".format(best[0], best[1], best[2]))


    x = list(gen_c(nc, 1, 20))
    colors = ["orangered", "darkblue", "limegreen", "dodgerblue", "violet"]
    fig, ax = plt.subplots()
    ax.set_xlabel("C")
    ax.set_ylabel("Accuracy")
    for i in range(len(kernels)):
        ax.plot(x, history[kernels[i]], linewidth=0.8, color=colors[i+1])
    ax.legend(labels=kernels)
    plt.show()

elif usedModel.lower() == "gpc":
    ### GPC

    # kernelsName = ["rbf", "matern", "exp sine quared", "rational quadratic"]
    # kernels = [gaussian_process.kernels.RBF(), gaussian_process.kernels.Matern(),
    #            gaussian_process.kernels.ExpSineSquared(), gaussian_process.kernels.RationalQuadratic()]

    # hist = {}
    # kernel = 1.0 * gaussian_process.kernels.RBF([1.0 for i in range(36)])
    # for i in range(len(kernels)):
    bestrandom = 0
    for j in range(10):
        xtrain, ytrain, xtest, ytest = split(types, data)
        ytrain, ytest = labelT(ytrain, ytest)
        base = gaussian_process.GaussianProcessClassifier(n_jobs=4)
        model = pipeline.make_pipeline(StandardScaler(), base)
        model.fit(xtrain, ytrain)
        acc = model.score(xtest, ytest)
        if acc > bestrandom:
            bestrandom = acc
        # hist[kernelsName[i]] = bestrandom

    print(bestrandom)

else:
    ### VC
    ## Need to change hyperparameters of every model

    bestrandom = 0
    for j in range(5):
        xtrain, ytrain, xtest, ytest = split(types, data)
        ytrain, ytest = labelT(ytrain, ytest)
        est1 = svm.SVC(kernel="rbf", C=6.32, probability=True) #
        est2 = gaussian_process.GaussianProcessClassifier(n_jobs=2)
        est3 = neighbors.KNeighborsClassifier(n_neighbors=4, weights="distance", metric="manhattan")
        base = ensemble.VotingClassifier(estimators=[("svc", est1), ("gpc", est2), ("knn", est3)], voting="soft", weights=[0.5, 0.2, 0.3])
        model = pipeline.make_pipeline(StandardScaler(), base)
        model.fit(xtrain, ytrain)
        acc = model.score(xtest, ytest)
        if acc > bestrandom:
            bestrandom = acc

    print("{:.2%}".format(bestrandom))