from Quantisations import SVMDecisionTreeQuantisation
from WhiteboxRNNCounterexampleGenerator import WhiteboxRNNCounterexampleGenerator
from time import clock

class Teacher:
    def __init__(self, network, num_dims_initial_split=10,starting_examples=None):
        if None is starting_examples:
            starting_examples = []
        self.recorded_words = {} # observation table uses this as its T (according to angluin paper terminology)
        self.discretiser = SVMDecisionTreeQuantisation(num_dims_initial_split)
        self.counterexample_generator = WhiteboxRNNCounterexampleGenerator(network,self.discretiser,starting_examples)
        self.dfas = []
        self.counterexamples_with_times = []
        self.current_ce_count = 0
        self.network = network
        self.alphabet = network.alphabet #this is more for intuitive use by lstar (it doesn't need to know there's a network involved)

    def update_words(self,words):
        seen = set(self.recorded_words.keys())
        words = set(words) - seen #need this to avoid answering same thing twice, which may happen a lot now with optimistic querying...
        self.recorded_words.update({w:self.network.classify_word(w) for w in words})

    def classify_word(self, w):
        return self.network.classify_word(w)

    def equivalence_query(self, dfa):
        self.dfas.append(dfa)
        start = clock()
        counterexample,message = self.counterexample_generator.counterexample(dfa)
        counterexample_time = clock() - start
        print(message)
        print("equivalence checking took: " + str(counterexample_time))
        if not None is counterexample:
            self.counterexamples_with_times.append((counterexample,counterexample_time))
            return counterexample
        return None