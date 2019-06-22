# Author : Hyunwoong
# When : 5/10/2019
# Homepage : github.com/gusdnd852
import os

import pandas as pd


def make_intent_data():
    result = pd.DataFrame(columns=['question', 'intent'])
    root = './data/'

    for filename in os.listdir(root):
        try:
            file = pd.read_csv(root + '/' + filename,
                               names=['question', 'intent'])
            result = pd.concat([result, file], sort=True)
        except:
            pass
    result = pd.DataFrame(result, columns=['question', 'intent'])
    result.to_csv('./intent/train_intent.csv', index=None)
    data = result
    intent_mapping = {}

    for q, i in data.values:
        intent_mapping[i] = 0

    for q, i in data.values:
        intent_mapping[i] += 1

    for i, x in enumerate(intent_mapping.items()):
        print(i, ' : ', x)
