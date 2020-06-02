"""
@author : Hyunwoong
@when : 5/11/2020
@homepage : https://github.com/gusdnd852
"""
import os
import pandas as pd


def build_intent(root):
    files = os.listdir(root)
    files = [pd.read_csv(root + file, encoding='utf-8') for file in files]
    concatenated = pd.concat(files, axis=0)
    concatenated.to_csv('total_intent.csv', index=False, header=['question', 'intent'])

if __name__ == '__main__':
    build_intent('data/intent/')
