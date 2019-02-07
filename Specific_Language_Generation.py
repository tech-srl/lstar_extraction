### hacky stuff in the name of sharing ###

import random
import string
from Training_Functions import make_train_set_for_target

bp_other_letters = string.ascii_lowercase  #probably avoid putting '$' in here because that's my dummy letter somewhere in the network (todo: something more general)
alphabet_bp = "()"+bp_other_letters


def make_similar(w, alphabet):
    new = list(w)
    indexes = list(range(len(new)))
    # switch characters
    num_switches = random.choice(range(3))
    shuffle(indexes)
    indexes_to_switch = indexes[:num_switches]
    for i in indexes_to_switch:
        new[i] = random.choice(alphabet)
    # insert characters
    num_inserts = random.choice(range(3))
    indexes = indexes + [len(new)]
    indexes_to_insert = indexes[:num_inserts]
    for i in indexes_to_insert:
        new = new[:i] + [random.choice(alphabet)] + new[i:]
    num_changes = num_switches + num_inserts
    # duplicate letters
    while ((num_changes == 0) or (random.choice(range(3)) == 0)) and len(new) > 0:
        index = random.choice(range(len(new)))
        new = new[:index + 1] + new[index:]
        num_changes += 1
    # omissions
    while ((num_changes == 0) or random.choice(range(3)) == 0) and len(new) > 0:
        index = random.choice(range(len(new)))
        new = new[:index] + new[index + 1:]
        num_changes +=1
    return ''.join(new)

def balanced_parantheses(w):
    open_counter = 0
    while len(w)>0:
        c = w[0]
        w = w[1:]
        if c == "(":
            open_counter += 1
        elif c==")":
            open_counter -= 1
            if open_counter <0:
                return False
    return open_counter == 0

def random_balanced_word(start_closing):
    count = 0
    word = ""
    while len(word)<start_closing:
        paran = (random.choice(range(3))==0)
        next_letter = random.choice("()") if paran else random.choice(bp_other_letters)
        if next_letter == ")" and count <= 0:
            continue
        word+=next_letter
        if next_letter == "(":
            count += 1
        if next_letter == ")":
            count -= 1
    while True:
        paran = (random.choice(range(3))==0)
        next_letter = random.choice(")") if paran else random.choice(bp_other_letters)
        if next_letter == ")":
            count -= 1
            if count<0:
                break
        word+=next_letter
    return word
        
def n_balanced_words_around_lengths(n,short,longg):
    words = set()
    while len(words)<n:
        for l in range(short,longg):
            words.add(random_balanced_word(l))
#     print('\n'.join(sorted(list(words),key=len)))
    return words

def get_balanced_parantheses_train_set(n,short,longg,lengths=None,max_train_samples_per_length=300,search_size_per_length=200): # eg 15000, 2, 30
    balanced_words = list(n_balanced_words_around_lengths(n,short,longg))
    almost_balanced = [make_similar(w,alphabet_bp) for w in balanced_words][:int(2*n/3)]
    less_balanced = [make_similar(w,alphabet_bp) for w in almost_balanced]
    barely_balanced = [make_similar(w,alphabet_bp) for w in less_balanced]

    all_words = balanced_words + almost_balanced + less_balanced + barely_balanced
    return make_train_set_for_target(balanced_parantheses,alphabet_bp,lengths=lengths,\
        max_train_samples_per_length=max_train_samples_per_length,\
        search_size_per_length=search_size_per_length,\
        provided_examples=all_words):