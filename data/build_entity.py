def check_entity_data(file):
    number_of_error = 0
    for i, data in enumerate(zip(file['question'].tolist(),
                                 file['entity'].tolist())):

        s = str(data[0]).split(' ')
        e = str(data[1]).split(' ')

        if len(s) != len(e):
            print(i, s, e)
            number_of_error += 1

    return number_of_error
