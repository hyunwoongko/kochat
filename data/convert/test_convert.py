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
    df = pd.read_csv('convert/testfile.csv', delimiter='|').values.tolist()
    df = [[i[0], '여행지'] for i in df]
    df = pd.DataFrame(df)
    df.to_csv('convert/testfile_convert.csv', header=['question', 'intent'], index=False)


if __name__ == '__main__':
    convert()
