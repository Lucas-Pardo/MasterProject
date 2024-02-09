
import numpy as np
import pandas as pd
from sklearn import svm, metrics, neighbors, multioutput, multiclass, pipeline, \
                    discriminant_analysis, kernel_ridge, gaussian_process, cross_decomposition, tree, ensemble, \
                    model_selection, calibration, neural_network
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from itertools import cycle

# Parameters
# usedModel = "knn"   # Model: {knn, svc, gpc, vc}

# Import data

data = pd.read_excel("Data/prediction_data1.xlsx")

types = {}

for x in data["P.Type"].values:
    if x in types:
        types[x] += 1
    else:
        types[x] = 1


# print(data.head())
print("Powder types set sizes:", types)

# Type classification

def split(pwd_types, features, feat_names, test_ratio=0.2):
    train = pd.DataFrame(columns=features.columns)
    test = pd.DataFrame(columns=features.columns)

    for c in pwd_types.keys():
        aux = features.loc[features["P.Type"] == c]
        tr, ts = train_test_split(aux, test_size=test_ratio)
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


def grid_search(xtrain, ytrain, xtest, ytest, model, parameters):
    gsearch = GridSearchCV(model, parameters, n_jobs=2, refit=True)
    gsearch.fit(xtrain, ytrain)
    print("Best score in CV: {:.2%}".format(gsearch.best_score_))
    print("Optimized prediction: {:.2%}".format(gsearch.score(xtest, ytest)))
    print("Best parameters:", gsearch.best_params_)
    return gsearch


def roc(ytrain, xtest, ytest, model, target_names):
    y_score = model.predict_proba(xtest)
    n_classes = y_score.shape[1]
    label_binarizer = LabelBinarizer().fit(ytrain)
    y_onehot_test = label_binarizer.transform(ytest)
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    fig, ax = plt.subplots(figsize=(6, 6))
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()


# Models

# if usedModel.lower() == "knn":
    ### KNN:
    # best = [0, 0, 0, 0, 0]
    # weight = ["uniform", "distance"]
    # metrics = ["euclidean", "manhattan", "chebyshev", "canberra", "braycurtis"]
    # history = {}
    # n_size = 10
    # xtrain, ytrain, xtest, ytest = split(types, data)
    # ytrain, ytest = labelT(ytrain, ytest)
    #
    # for metric in metrics:
    #     hist = np.zeros((2, n_size))
    #     for w in range(len(weight)):
    #         for n in range(1, n_size+1):
    #             base = neighbors.KNeighborsClassifier(n_neighbors=n, weights=weight[w], metric=metric)
    #             model = pipeline.make_pipeline(StandardScaler(), base)
    #             scores = cross_val_score(model, xtrain, ytrain, cv=3)
    #             smean = scores.mean()
    #             hist[w, n-1] = smean
    #             if smean > best[3]:
    #                 best = [metric, weight[w], n, smean, scores.std()]
    #     history[metric] = hist
    #
    # print("Best in cross validation: (Metric: {}, Weights: {}, Neighbors: {}, acc: {:.2%}+-{:.1%})".format(best[0], best[1], best[2], best[3], best[4]))
    # base = neighbors.KNeighborsClassifier(n_neighbors=best[2], weights=best[1], metric=best[0])
    # model = pipeline.make_pipeline(StandardScaler(), base)
    # model.fit(xtrain, ytrain)
    # acc = model.score(xtest, ytest)
    # print("Final prediction: {:.2%}".format(acc))
    # # base = neighbors.KNeighborsClassifier(n_neighbors=2, weights="distance", metric="euclidean")
    # # model = pipeline.make_pipeline(StandardScaler(), base)
    # # model.fit(xtrain, ytrain)
    # # acc = model.score(xtest, ytest)
    # # print("Final prediction: {:.2%}".format(acc))
    #
    # importance = permutation_importance(model, xtrain, ytrain, n_repeats=5).importances_mean
    # feat_names = ["P.Conc", "Voltage", "Mt frac", "Radius avg", "Depth avg", "Sa avg"]
    # fig, ax = plt.subplots()
    # x_pos = np.arange(start=1, stop=len(feat_names)+1)
    # ax.set_xticks(x_pos, labels=feat_names, rotation=30)
    # ax.bar(x_pos, importance, width=0.2)
    # plt.show()
    #
    # x_pos = np.arange(start=1, stop=n_size+1)
    # relPos = [-0.2, -0.1, 0, 0.1, 0.2]
    # colors = ["orangered", "gold", "limegreen", "dodgerblue", "violet"]
    # for w in range(len(weight)):
    #     fig, ax = plt.subplots()
    #     ax.set_xticks(x_pos, labels=x_pos, rotation=0)
    #     ax.set_xlabel("Neighbors")
    #     ax.set_ylabel("Accuracy")
    #     ax.set_title("Weights: " + weight[w])
    #     for i in range(5):
    #         ax.bar(x_pos+relPos[i], history[metrics[i]][w, :], width=0.1, color=colors[i])
    #     ax.legend(labels=metrics)
    #     plt.show()

### Logistic regression
# xtrain, ytrain, xtest, ytest = split(types, data)
# ytrain, ytest = labelT(ytrain, ytest)
# base = LogisticRegression()
# model = pipeline.make_pipeline(StandardScaler(), base)
# model.fit(xtrain, ytrain)
# acc = model.score(xtest, ytest)
# print("Final prediction: {:.2%}".format(acc))
# parameters = {"logisticregression__C": np.arange(1, 11), "logisticregression__class_weight": [None, "balanced"],
#               "logisticregression__solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]}
# gsearch = GridSearchCV(model, parameters, n_jobs=2, refit=True)
# gsearch.fit(xtrain, ytrain)
# print("Best score: {:.2%}".format(gsearch.best_score_))
# print(gsearch.best_params_)
# print("Prediction score: {:.2%}".format(gsearch.score(xtest, ytest)))

### Decision tree
# xtrain, ytrain, xtest, ytest = split(types, data)
# ytrain, ytest = labelT(ytrain, ytest)
# base = tree.DecisionTreeClassifier()
# model = pipeline.make_pipeline(StandardScaler(), base)
# model.fit(xtrain, ytrain)
# acc = model.score(xtest, ytest)
# print("Final prediction: {:.2%}".format(acc))
# parameters = {"decisiontreeclassifier__class_weight": [None, "balanced"],
#             "decisiontreeclassifier__criterion": ["gini", "entropy", "log_loss"]}
# gsearch = GridSearchCV(model, parameters, n_jobs=2, refit=True)
# gsearch.fit(xtrain, ytrain)
# print("Best score: {:.2%}".format(gsearch.best_score_))
# print(gsearch.best_params_)
# print("Prediction score: {:.2%}".format(gsearch.score(xtest, ytest)))

### Random forest
ts = 0.3
# feat_names = ["P.Conc", "Voltage", "Mt frac", "Radius avg", "Depth avg", "Sa avg"]
feat_names = ["P.Conc", "Mt frac", "Radius avg", "Depth avg", "Sa avg"]
xtrain, ytrain, xtest, ytest = split(types, data, feat_names, ts)
ytrain, ytest = labelT(ytrain, ytest)
print("Train set size:", len(ytrain), "({:.0%})".format(1-ts))
print("Test set size:", len(ytest), "({:.0%})".format(ts))
base = ensemble.RandomForestClassifier()
model = pipeline.make_pipeline(StandardScaler(), base)
model.fit(xtrain, ytrain)
acc = model.score(xtest, ytest)
print("Unoptimized prediction: {:.2%}".format(acc))
parameters = {"randomforestclassifier__class_weight": [None, "balanced", "balanced_subsample"],
              "randomforestclassifier__criterion": ["gini", "entropy", "log_loss"],
              "randomforestclassifier__n_estimators": list(range(10, 200, 10))}
gsearch = grid_search(xtrain, ytrain, xtest, ytest, model, parameters)
roc(ytrain, xtest, ytest, gsearch, list(types.keys()))
importance = permutation_importance(gsearch, xtrain, ytrain, n_repeats=10, n_jobs=2)
importance_mean = pd.Series(importance.importances_mean, index=feat_names)
fig, ax = plt.subplots()
importance_mean.plot.bar(yerr=importance.importances_std, ax=ax, rot=30)
ax.set_ylabel("Mean accuracy decrease")
# x_pos = np.arange(start=1, stop=len(feat_names)+1)
# ax.set_xticks(x_pos, labels=feat_names, rotation=30)
# ax.bar(x_pos, importance, width=0.2)
plt.title("Feature importances using permutation")
plt.show()

# elif usedModel.lower() == "svc":
    ### SVC
    # best = [0, 0, 0]
    # kernels = ["linear", "poly", "rbf"]#, "sigmoid"]
    # history = {}
    # nc = 50
    #
    # for kernel in kernels:
    #     hist = np.zeros((nc,))
    #     k = 0
    #     for c in gen_c(nc, 1, 20):
    #         bestrandom = 0
    #         for i in range(5):
    #             xtrain, ytrain, xtest, ytest = split(types, data)
    #             ytrain, ytest = labelT(ytrain, ytest)
    #             base = svm.SVC(kernel=kernel, C=c)
    #             model = pipeline.make_pipeline(StandardScaler(), base)
    #             model.fit(xtrain, ytrain)
    #             acc = model.score(xtest, ytest)
    #             if acc > bestrandom:
    #                 bestrandom = acc
    #         hist[k] = bestrandom
    #         if bestrandom > best[2]:
    #             best = [kernel, c, bestrandom]
    #         k += 1
    #         print(kernel, k/nc)
    #     history[kernel] = hist
    #
    #
    # print("Best: (Kernel: {}, C: {}, acc: {:.2%})".format(best[0], best[1], best[2]))
    #
    #
    # x = list(gen_c(nc, 1, 20))
    # colors = ["orangered", "darkblue", "limegreen", "dodgerblue", "violet"]
    # fig, ax = plt.subplots()
    # ax.set_xlabel("C")
    # ax.set_ylabel("Accuracy")
    # for i in range(len(kernels)):
    #     ax.plot(x, history[kernels[i]], linewidth=0.8, color=colors[i+1])
    # ax.legend(labels=kernels)
    # plt.show()

# elif usedModel.lower() == "gpc":
    ### GPC

    # # kernelsName = ["rbf", "matern", "exp sine quared", "rational quadratic"]
    # # kernels = [gaussian_process.kernels.RBF(), gaussian_process.kernels.Matern(),
    # #            gaussian_process.kernels.ExpSineSquared(), gaussian_process.kernels.RationalQuadratic()]
    #
    # # hist = {}
    # # kernel = 1.0 * gaussian_process.kernels.RBF([1.0 for i in range(36)])
    # # for i in range(len(kernels)):
    # bestrandom = 0
    # for j in range(10):
    #     xtrain, ytrain, xtest, ytest = split(types, data)
    #     ytrain, ytest = labelT(ytrain, ytest)
    #     base = gaussian_process.GaussianProcessClassifier(n_jobs=4)
    #     model = pipeline.make_pipeline(StandardScaler(), base)
    #     model.fit(xtrain, ytrain)
    #     acc = model.score(xtest, ytest)
    #     if acc > bestrandom:
    #         bestrandom = acc
    #     # hist[kernelsName[i]] = bestrandom
    #
    # print(bestrandom)

# else:
    ### VC
    ## Need to change hyperparameters of every model

    # bestrandom = 0
    # for j in range(5):
    #     xtrain, ytrain, xtest, ytest = split(types, data)
    #     ytrain, ytest = labelT(ytrain, ytest)
    #     est1 = svm.SVC(kernel="rbf", C=6.32, probability=True) #
    #     est2 = gaussian_process.GaussianProcessClassifier(n_jobs=2)
    #     est3 = neighbors.KNeighborsClassifier(n_neighbors=4, weights="distance", metric="manhattan")
    #     base = ensemble.VotingClassifier(estimators=[("svc", est1), ("gpc", est2), ("knn", est3)], voting="soft", weights=[0.5, 0.2, 0.3])
    #     model = pipeline.make_pipeline(StandardScaler(), base)
    #     model.fit(xtrain, ytrain)
    #     acc = model.score(xtest, ytest)
    #     if acc > bestrandom:
    #         bestrandom = acc
    #
    # print("{:.2%}".format(bestrandom))