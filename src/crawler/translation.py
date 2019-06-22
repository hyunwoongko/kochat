from googletrans import Translator

translator = Translator()


def translate(text, lang, src='auto'):
    return translator.translate(text, src=src, dest=lang).text
