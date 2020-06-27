from _dialogue import config


def crawler(cls):
    for key, val in config.CRAWLER.items():
        setattr(cls, key, val)
    return cls


def searcher(cls):
    cls = crawler(cls)
    for key, val in config.SEARCH.items():
        setattr(cls, key, val)

    return cls


def editor(cls):
    cls = crawler(cls)
    for key, val in config.EDIT.items():
        setattr(cls, key, val)

    return cls


def answerer(cls):
    cls = crawler(cls)
    for key, val in config.ANSWER.items():
        setattr(cls, key, val)
    return cls
