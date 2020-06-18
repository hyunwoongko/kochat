from sklearn.tree import \
    DecisionTreeClassifier, \
    ExtraTreeClassifier
from sklearn.ensemble import \
    RandomForestClassifier, \
    AdaBoostClassifier, \
    BaggingClassifier, \
    ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import \
    LogisticRegression, \
    LogisticRegressionCV, \
    SGDClassifier, \
    LinearRegression
from sklearn.naive_bayes import \
    GaussianNB, \
    BernoulliNB
from sklearn.neighbors import \
    KNeighborsClassifier
from sklearn.neural_network import \
    MLPClassifier
from sklearn.svm import \
    LinearSVC, \
    SVC, \
    NuSVC

from backend.decorators import model

SKLEARN_ALL_MODELS = []
for _, obj in globals().copy().items():
    if isinstance(obj, type):
        m = model(obj)()

        if hasattr(m, 'max_iter'):
            setattr(m, 'max_iter', 1500)

        SKLEARN_ALL_MODELS.append(m)
