from konlpy.tag import Okt

stop_word = [
]

josa = [
    '이구나', '이네', '이야',
    '은', '는', '이', '가', '을', '를',
    '로', '으로', '이야', '야', '냐', '니'
]


def tokenize(sentence):
    tokenizer = Okt()
    word_bag = []
    pos = tokenizer.pos(sentence)
    for word, tag in pos:
        if word in stop_word:
            continue
        elif (tag == 'Josa' and word in josa) or tag == 'Punctuation':
            continue
        else:
            word_bag.append(word)
    result = ''.join(word_bag)
    return result
