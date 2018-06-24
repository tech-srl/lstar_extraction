def tomita_1(word):
    return not "0" in word

def tomita_2(word):
    return word=="10"*(int(len(word)/2))

import re
_not_tomita_3 = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$") 
# *not* tomita 3: words containing an odd series of consecutive ones and then later an odd series of consecutive zeros
# tomita 3: opposite of that
def tomita_3(w): 
    return None == _not_tomita_3.match(w) #complement of _not_tomita_3

def tomita_4(word):
    return not "000" in word

def tomita_5(word):
    return (word.count("0")%2 == 0) and (word.count("1")%2 == 0)

def tomita_6(word):
    return ((word.count("0")-word.count("1"))%5) == 0

def tomita_7(word):
    return word.count("10") <= 1