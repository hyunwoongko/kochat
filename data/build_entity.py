import os
import pandas as pd


def check_entity_data(file):
    number_of_error = 0
    for i, data in enumerate(zip(file['question'].tolist(),
                                 file['label'].tolist())):

        s = str(data[0]).split(' ')
        e = str(data[1]).split(' ')

        if len(s) != len(e):
            print(i, s, e)
            number_of_error += 1

    if number_of_error != 0:
        raise Exception("THERE ARE {} ERRORS!\n".format(number_of_error))

    return number_of_error


def build_intent(root):
    files = os.listdir(root)
    for file_name in files:
        entity_file = pd.read_csv(root + file_name, encoding='utf-8')
        check_entity_data(entity_file)


if __name__ == '__main__':
    build_intent('raw/')
