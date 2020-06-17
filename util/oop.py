"""
@author : Hyunwoong
@when : 6/16/2020
@homepage : https://github.com/gusdnd852

파이썬으로 OOP 해보자 ^___^
"""


def global_var(**kwargs):
    """
    데코레이터 파라미터로 적으면 전역변수로 쓸수 있음
    예시로 바로 아래 오버로드 데코레이터에서 사용함
    """

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@global_var(func_table={})
def overload(*args):
    """
    함수를 오버로딩할 수 있는 데코레이터
    (모듈이름, 함수이름, 함수타입) => "호출 시" 함수를 구분하는 요소
    정규화된 이름 (qualname)사용으로 클래스 구분도 가능함
    """

    def overload_wrapper(func):
        func_key = func.__module__, func.__qualname__, args
        overload.func_table[func_key] = func

        def call_by_signature(*args_):
            key = func.__module__, func.__qualname__, \
                  tuple([type(i) for i in args_])

            find = overload.func_table[key]
            return find(*args_)

        # 정보 싹 세팅
        call_by_signature.__doc__ = func.__doc__
        call_by_signature.__name__ = func.__name__
        call_by_signature.__qualname__ = func.__qualname__
        call_by_signature.__defaults__ = func.__defaults__
        call_by_signature.__annotations__ = func.__annotations__
        call_by_signature.__module__ = func.__module__
        return call_by_signature

    return overload_wrapper


def override(super_cls):
    """
    이 함수 상속 받은 거에요!"하고 말하기 위해 사용
    실제로 오버라이딩 하고있는지 검사해서 assert함
    """

    def overrider(method):
        assert (method.__name__ in dir(super_cls))
        return method

    return overrider


def singleton(class_):
    """
    싱글톤 객체 생성을 위한 데코레이터
    """

    class class_w(class_):
        _instance = None

        def __new__(class_, *args, **kwargs):
            if class_w._instance is None:
                class_w._instance = super(class_w, class_) \
                    .__new__(class_, *args, **kwargs)
                class_w._instance._sealed = False
            return class_w._instance

        def __init__(self, *args, **kwargs):
            if self._sealed:
                return
            super(class_w, self).__init__(*args, **kwargs)
            self._sealed = True

    class_w.__name__ = class_.__name__
    return class_w
