from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def accuracy_score(y, y_pred):
    num_test = len(y)
    correct = 0.
    for idx in range(num_test):
        if y_pred[idx] == y[idx]:
            correct += 1
    return correct/num_test


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def getLemmatizedDocument(doc):
    en_stop = set(stopwords.words('english'))
    en_stop.add('.')
    en_stop.add(')')
    en_stop.add('(')
    en_stop.add(']')
    en_stop.add('[')
    en_stop.add("'")

    tokens = word_tokenize(doc.lower())
    stopped_tokens = filter(lambda token: token not in en_stop, tokens)
    stopped_tokens = list(stopped_tokens)

    lemmatizer = WordNetLemmatizer()
    doc_a = [lemmatizer.lemmatize(word, pos='a') for word in stopped_tokens]
    doc_n = [lemmatizer.lemmatize(word, pos='n') for word in doc_a]

    return doc_n
