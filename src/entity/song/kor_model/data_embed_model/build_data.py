from src.entity.song.kor_model import CoNLLDataset
from src.entity.song.kor_model.data_embed_model import word2vec, data_utils


def build_data(config):
    """
    Procedure to build data
    """
    # build embedding model
    embed_model = word2vec.train_w2v(config)

    #data_utils.write_vocab(model.wv.index2word, config.words_filename)
    #np.savetxt(config.trimmed_filename, model.wv.syn0)

    # Generators
    dev   = CoNLLDataset(config.dev_filename, max_iter=config.max_iter)
    test  = CoNLLDataset(config.test_filename, max_iter=config.max_iter)
    train = CoNLLDataset(config.train_filename, max_iter=config.max_iter)

    vocab_words, vocab_tags = data_utils.get_vocabs([train, dev, test])

    vocab = vocab_words & set(embed_model.wv.index2word)
    vocab.add(data_utils.UNK)
    # vocab.add(data_utils.NUM)

    vocab_chars = data_utils.get_char_vocab(train)

    data_utils.write_char_embedding(vocab_chars, config.charembed_filename)
    data_utils.write_vocab(vocab_chars, config.chars_filename)
    data_utils.write_vocab(vocab, config.words_filename)
    data_utils.write_vocab(vocab_tags, config.tags_filename)

    data_utils.export_trimmed_glove_vectors(vocab, embed_model, config.trimmed_filename)
