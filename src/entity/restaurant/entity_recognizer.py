import tensorflow as tf

from src.entity.restaurant.kor_model.config import config
from src.entity.restaurant.kor_model.data_embed_model import word2vec, data_utils
from src.entity.restaurant.kor_model.data_embed_model.data_utils import CoNLLDataset
from src.entity.restaurant.kor_model.ner_model.lstmcrf_model import NERModel


def embed_model():
    return word2vec.train_w2v(config)


def data_iterator():
    dev = CoNLLDataset(config.dev_filename, max_iter=config.max_iter)
    test = CoNLLDataset(config.test_filename, max_iter=config.max_iter)
    train = CoNLLDataset(config.train_filename, max_iter=config.max_iter)
    return [train, dev, test]


# Data Set 에서 Word 와 Tag Distinct Value 를 추출
def get_vocab():
    vocab_words, vocab_tags = data_utils.get_vocabs(data_iterator())
    vocab = vocab_words & set(embed_model().wv.index2word)
    vocab.add(data_utils.UNK)
    vocab_chars = data_utils.get_char_vocab(data_iterator()[0])
    data_utils.write_char_embedding(vocab_chars, config.charembed_filename)
    data_utils.write_vocab(vocab_chars, config.chars_filename)
    data_utils.write_vocab(vocab, config.words_filename)
    data_utils.write_vocab(vocab_tags, config.tags_filename)
    data_utils.export_trimmed_glove_vectors(vocab, embed_model, config.trimmed_filename)
    return vocab_words, vocab_tags, vocab_chars


def get_embeddings():
    return data_utils.get_trimmed_glove_vectors(config.trimmed_filename)


def get_char_embedding():
    return data_utils.get_trimmed_glove_vectors(config.charembed_filename)


def get_vocab_words():
    return data_utils.load_vocab(config.words_filename)


def get_vocab_tags():
    return data_utils.load_vocab(config.tags_filename)


def get_vocab_chars():
    return data_utils.load_vocab(config.chars_filename)


def get_processing_word():
    return data_utils.get_processing_word(get_vocab_words(),
                                          get_vocab_chars(),
                                          lowercase=config.lowercase,
                                          chars=config.chars)


def get_processing_tag():
    return data_utils.get_processing_word(get_vocab_tags(),
                                          lowercase=False)


def get_dev():
    return CoNLLDataset(config.dev_filename, get_processing_word(), get_processing_tag(), config.max_iter)


def get_test():
    return CoNLLDataset(config.test_filename, get_processing_word(), get_processing_tag(), config.max_iter)


def get_train():
    return CoNLLDataset(config.train_filename, get_processing_word(), get_processing_tag(), config.max_iter)


def get_restaurant_entity(sentence, is_train):
    if is_train:
        get_vocab()
        model = NERModel(config, get_embeddings(), ntags=len(get_vocab_tags()), nchars=len(get_vocab_chars()),
                         logger=None,
                         char_embed=get_char_embedding())
        model.build()
        model.train(get_train(), get_dev(), get_vocab_tags())
        model.evaluate(get_test(), get_vocab_tags())
        return model.predict(get_vocab_tags(), get_processing_word(), sentence)
    else:
        tf.reset_default_graph()
        model = NERModel(config, get_embeddings(), ntags=len(get_vocab_tags()), nchars=len(get_vocab_chars()),
                         logger=None,
                         char_embed=get_char_embedding())
        model.build()
        return model.predict(get_vocab_tags(), get_processing_word(), sentence)
