import pandas as pd
import re


def check_entity_data(q, e):
    number_of_error = 0
    for i, data in enumerate(zip(q, e)):

        s = str(data[0]).split(' ')
        e = str(data[1]).split(' ')

        if len(s) != len(e):
            print(i, s, e)
            number_of_error += 1

    return number_of_error


def convert():
    f = pd.read_csv('data/convert/testfile.csv', delimiter=',', encoding='utf-8')
    question = [re.sub('\t', '', s) for s in f['question'].tolist()]
    entity = []
    for i, e in enumerate(f['entity'].tolist()):
        while '\t\t' in e:
            e = re.sub('\t\t', '\t', str(e))
            print(i)

        if e[len(e) - 1] == '\t':
            e = e[0:len(e) - 1]

        e = re.sub('\t', ' ', e)
        entity.append(e)

    if check_entity_data(question, entity) != 0:
        raise Exception("number of token and entity must be same !")

    f = pd.DataFrame(data=zip(question, entity))
    f.to_csv('data/convert/testfile_convert.csv', index=False)


if __name__ == '__main__':
    convert()
