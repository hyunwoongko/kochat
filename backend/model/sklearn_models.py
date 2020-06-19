from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from backend.decorators import model

LINEAR_MODELS = []
for _, m in globals().copy().items():
    if isinstance(m, type):

        if hasattr(m, 'max_iter'):
            setattr(m, 'max_iter', 5000)

        # 모듈 경로 직접 잡아서 저장 경로 맞춰야 함
        m.__module__ = __file__.replace('/', '.')
        m.__module__ = m.__module__.split('.')
        m.__module__ = m.__module__[len(m.__module__) - 4: len(m.__module__) - 1]
        m.__module__ = '.'.join(m.__module__)
        m = model(m)()

        LINEAR_MODELS.append(m)
