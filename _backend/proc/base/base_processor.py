from abc import abstractmethod, ABCMeta

from sklearn.metrics import \
    accuracy_score, \
    confusion_matrix, \
    classification_report

from _backend.decorators import proc


@proc
class BaseProcessor(metaclass=ABCMeta):
    def __init__(self, model):
        """
        모든 프로세서의 부모클래스입니다.
        모델들의 이름과 파일 주소를 가지고 있고, 다양한 추상 메소드를 가지고 있습니다.
        학습 및 검증 등의 메소드의 이름은 sklearn과 동일하게 설정했습니다.

        :param model: 학습할 모델
        """

        super().__init__()
        self.train_data, self.test_data = None, None
        self.model = model
        self.model_loaded = False

        self.model_dir = self.model_dir + \
                         self.__class__.__name__ + \
                         self.delimeter

        self.model_file: str = self.model_dir + \
                               self.delimeter + \
                               self.__class__.__name__

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        학습을 마친 모델이 유저의 입력에 inference 할때 사용됩니다.
        """

        raise NotImplementedError

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        모델을 학습합니다.
        """

        raise NotImplementedError

    @abstractmethod
    def test(self, *args, **kwargs):
        """
        모델을 검증합니다.
        """

        raise NotImplementedError

    @abstractmethod
    def _load_model(self):
        """
        저장된 모델을 불러옵니다.
        """

        raise NotImplementedError

    @abstractmethod
    def _save_model(self):
        """
        모델을 저장장치에 저장합니다.
        """

        raise NotImplementedError

    def _accuracy(self, label, predict) -> float:
        """
        분류 정확도를 출력합니다.

        :param label: 라벨
        :param predict: 예측
        :return: 정확도
        """

        return accuracy_score(label, predict)

    def _classification_report(self, label, predict):
        """
        분류 보고서를 출력합니다.
        여기에는 Precision, Recall, F1 Score, Accuracy 등이 포함됩니다.

        :param label: 라벨
        :param predict: 예측
        :return: 다양한 메트릭으로 측정한 모델 성능
        """

        return classification_report(label, predict)

    def _confusion_matrix(self, label, predict):
        """
        컨퓨전 매트릭스를 출력합니다.

        :param label: 라벨
        :param predict: 예측
        :return: 컨퓨전 매트릭스
        """

        return confusion_matrix(label, predict)

    def _print_log(self, msg):
        """
        로깅시 클래스 이름을 항상 붙여서 출력합니다.

        :param msg: 출력할 메시지
        """

        print('{name} - {msg}'.format(name=__class__.__name__, msg=msg))
