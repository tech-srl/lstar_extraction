from LSTM import LSTMNetwork
from GRU import GRUNetwork
from LinearTransform import LinearTransform
import dynet as dy
from time import clock
import random
import matplotlib.pyplot as plt
from math import ceil

class RNNClassifier:
    def __init__(self,alphabet,num_layers=2,input_dim=3,hidden_dim=5,RNNClass=LSTMNetwork):

        self.alphabet = list(alphabet)
        self.int2char = self.alphabet
        self.char2int = {c:i for i,c in enumerate(self.int2char)}
        self.int2class = [True,False] # binary classifier for now
        self.class2int = {c:i for i,c in enumerate(self.int2class)}
        self.vocab_size = len(self.alphabet)
        
        self.pc = dy.ParameterCollection()
        self.lookup = self.pc.add_lookup_parameters((self.vocab_size, input_dim))
        self.linear_transform = LinearTransform(hidden_dim,len(self.class2int),self.pc)
        self.rnn = RNNClass(num_layers=num_layers,input_dim=input_dim,hidden_dim=hidden_dim,pc=self.pc)
        self.store_expressions() 
        self.all_losses = []
        self.finish_signal = "Finished"
        self.keep_going = "Keep Going"


    def renew(self):
        dy.renew_cg()
        self.store_expressions()

    def store_expressions(self):
        self.rnn.store_expressions()
        self.linear_transform.store_expressions()
            
    def _char_to_input_vector(self,char):
        return self.lookup[self.char2int[char]]
            
    def _next_state(self,state,char):
        return self.rnn.next_state(state,self._char_to_input_vector(char))
    
    def _state_probability_distribution(self,state):
        return dy.softmax(self.linear_transform.apply(state.output())) 
        
    def get_first_RState(self):
        return self.rnn.initial_state.as_vec(), self._classify_state(self.rnn.initial_state)
    
    def get_next_RState(self,vec,char): 
        #verification, could get rid of
        if not char in self.alphabet:
            print("char for next vector not from input alphabet")
            return None     
        state = self.rnn.state_class(full_vec = vec, hidden_dim = self.rnn.hidden_dim)
        state = self._next_state(state,char)
        return state.as_vec(), self._classify_state(state)
        
    def _word_is_over_input_alphabet(self,word):
        return next((False for c in word if not c in self.alphabet),True)
 
    def _state_accept_probability(self,s):
        probabilities = self._state_probability_distribution(s)
        return probabilities[self.class2int[True]]

    def _classify_state(self,s):
        return self._state_accept_probability(s).value()>0.5

    def _probability_word_in_language(self,word):
        #verification, could get rid of
        if not self._word_is_over_input_alphabet(word):
            print("word is not over input alphabet")
            return False
        s = self.rnn.initial_state
        for c in word:
            s = self._next_state(s,c)
        return self._state_accept_probability(s)

    def classify_word(self,word):
        return self._probability_word_in_language(word).value()>0.5

    def loss_on_word(self, word, label):
        s = self.rnn.initial_state
        p = self._probability_word_in_language(word)
        p = p if label == True else (1-p) # now p = probability of correct label for word
        #dy.picklneglogsoftmax on self.linear_transform.apply(state.output()) should be numerically stable
        return -dy.log(p) # ideally p should be 1, in which case log(p)=0. the lower it gets: the greater -log(p) gets
        # loss = dy.esum(loss) 
    
    def train_batch(self,word_dict,trainer):
        self.renew()
        loss = [self.loss_on_word(w,word_dict[w]) for w in word_dict]
        loss = dy.esum(loss)
        loss_value = loss.value()/len(word_dict)
        loss.backward()
        trainer.update()
        return loss_value

    def show_all_losses(self):
        plt.scatter(range(len(self.all_losses)),self.all_losses,label="classification loss since initiation")
        plt.legend()
        plt.show()

    def train_group(self,word_dict,iterations,trainer_class=dy.AdamTrainer,learning_rate=None,loss_every=100,
                    batch_size=20,show=True,print_time=True,stop_threshold=0):
        if iterations == 0:
            return
        start = clock()
        trainer = trainer_class(self.pc)
        if not None is learning_rate:
            trainer.learning_rate = learning_rate
        loss_values = []

        if None is batch_size:
            batch_size = len(word_dict) # leave None to define one huge batch

        words = list(word_dict.keys())
        num_batches = int(ceil(len(words)/batch_size))
        for i in range(iterations):
            random.shuffle(words)
            batches_loss = []
            for j in range(num_batches):
                batch = words[j*batch_size:(j+1)*batch_size]
                batches_loss.append(self.train_batch({w:word_dict[w] for w in batch},trainer))
            loss_values.append(sum(batches_loss)/num_batches) # its not perfect because the last batch might be a different size and they were training during, but whatever, its here to give a general idea of what's going on
            if loss_values[-1]<stop_threshold:
                break
            if (i+1)%loss_every == 0:
                print("current average loss is: ",loss_values[-1])

        self.all_losses += loss_values
        if print_time:
            print("total time:",clock()-start)
        if show:
            plt.scatter(range(len(loss_values)),loss_values,label="classification loss for these epochs")
            plt.legend()
            plt.show()
            self.show_all_losses()
        return self.finish_signal if loss_values[-1] < stop_threshold else self.keep_going