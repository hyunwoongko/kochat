from abc import abstractmethod
from typing import Any

from kochat.decorators import proc


@proc
class BaseProcessor:
    def __init__(self, model: Any):
        """
        모든 프로세서의 부모클래스입니다.
        모델들의 이름과 파일 주소를 가지고 있고, 다양한 추상 메소드를 가지고 있습니다.
        학습 및 검증 등의 메소드의 이름은 sklearn과 동일하게 설정했습니다.

        :param model: 학습할 모델
        """

        super().__init__()
        self.train_data, self.test_data = None, None
        self.ood_train, self.ood_test = None, None
        self.model = model
        self.model_loaded = False

        # /saved/CLASS_NAME/
        self.model_dir = self.model_dir + \
                         self.__class__.__name__ + \
                         self.delimeter

        # /saved/CLASS_NAME/CLASS_NAME.xxx
        self.model_file = self.model_dir + \
                          self.__class__.__name__

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _load_model(self):
        raise NotImplementedError

    @abstractmethod
    def _save_model(self):
        raise NotImplementedError

    def _print(self, msg: str, name: str = None):
        """
        Processor는 내용 출력시 반드시 자신의 이름을 출력해야합니다.

        :param msg: 출력할 메시지
        :return: [XXProcessor] message
        """

        if name is None:
            name = self.__class__.__name__

        print('[{name}] {msg}'.format(name=name, msg=msg))
