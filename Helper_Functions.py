import itertools
import random

def mean(num_list):
    return sum(num_list)*1.0/len(num_list)

def n_words_of_length(n,length,alphabet):
    if 50*n >= pow(len(alphabet),length):
        res = all_words_of_length(length, alphabet)
        random.shuffle(res)
        return res[:n]
    #else if 50*n < total words to be found, i.e. looking for 1/50th of the words or less
    res = set()
    while len(res)<n:
        word = ""
        for _ in range(length):
            word += random.choice(alphabet)
        res.add(word)
    return list(res)

def all_words_of_length(length,alphabet):
    return [''.join(list(b)) for b in itertools.product(alphabet, repeat=length)]


def compare(network,classifier,length,num_examples=1000,provided_samples=None):
    if not None is provided_samples:
        words = provided_samples
    else:
        words = n_words_of_length(num_examples,length,network.alphabet)
    disagreeing_words = [w for w in words if not (network.classify_word(w) == classifier.classify_word(w))]
    return 1-(len(disagreeing_words)/len(words)), disagreeing_words

def map_nested_dict(d,mapper):
    if not isinstance(d,dict):
        return mapper(d)
    return {k:map_nested_dict(d[k],mapper) for k in d}

class MissingInput(Exception):
    pass