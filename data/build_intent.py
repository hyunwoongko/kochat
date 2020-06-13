"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
import os
import pandas as pd
from pandas import DataFrame


def build_intent(root):
    files = os.listdir(root)
    intent_files = []
    for file_name in files:
        intent = file_name.split('.')[0]
        intent_file = pd.read_csv(root + file_name, encoding='utf-8')
        question = intent_file['question'].values.tolist()
        intent_file = [(data, intent) for data in question]
        intent_files += intent_file

    intent_files = DataFrame(intent_files)
    intent_files.to_csv('intent_data.csv', index=False, header=['question', 'intent'])


if __name__ == '__main__':
    build_intent('raw/')
