import os

from gensim.models import fasttext

def train_w2v(config):
    try:
        print("word2vec train start")
        update_flag = False
        model = fasttext.FastText(size=300, window=5, min_count=1, workers=4)

        with open(config.pos_path) as f:
            for line in f.readlines():
                if update_flag == False:
                    model.build_vocab([line.split(' ')], update=False)
                    update_flag = True
                else:
                    model.build_vocab([line.split(' ')], update=True)

        with open(config.pos_path) as f:

            for line in f.readlines():
                for _ in range(100):
                    model.train(line.split(' '), total_examples=model.corpus_count, epochs=model.epochs)

        os.makedirs(config.embedding_model_path, exist_ok=True)
        model.save(''.join([config.embedding_model_path, '/', 'model']))
        return model

    except Exception as e:
        print(Exception("error on train w2v : {0}".format(e)))
    finally:
        print("word2vec train done")
