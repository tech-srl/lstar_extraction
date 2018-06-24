import graphviz as gv
from IPython.display import Image
from IPython.display import display
import functools
from copy import deepcopy, copy
import itertools
import Lstar
from random import randint, shuffle
import random
from time import clock
import string

digraph = functools.partial(gv.Digraph, format='png')
graph = functools.partial(gv.Graph, format='png')

separator = "_"

class DFA:
    def __init__(self,obs_table):
        self.alphabet = obs_table.A #alphabet
        self.Q = [s for s in obs_table.S if s==obs_table.minimum_matching_row(s)] #avoid duplicate states
        self.q0 = obs_table.minimum_matching_row("")
        self.F = [s for s in self.Q if obs_table.T[s]== 1]
        self._make_transition_function(obs_table)

    def _make_transition_function(self,obs_table):
        self.delta = {}
        for s in self.Q:
            self.delta[s] = {}
            for a in self.alphabet:
                self.delta[s][a] = obs_table.minimum_matching_row(s+a)

    def classify_word(self,word):
        #assumes word is string with only letters in alphabet
        q = self.q0
        for a in word:
            q = self.delta[q][a]
        return q in self.F

    def draw_nicely(self,force=False,maximum=60): #todo: if two edges are identical except for letter, merge them and note both the letters
        if (not force) and len(self.Q) > maximum:
            return

        #suspicion: graphviz may be upset by certain sequences, avoid them in nodes
        label_to_number_dict = {False:0} #false is never a label but gets us started
        def label_to_numberlabel(label):
            max_number = max(label_to_number_dict[l] for l in label_to_number_dict)
            if not label in label_to_number_dict:
                label_to_number_dict[label] = max_number + 1
            return str(label_to_number_dict[label])

        def add_nodes(graph, nodes): #stolen from http://matthiaseisen.com/articles/graphviz/
            for n in nodes:
                if isinstance(n, tuple):
                    graph.node(n[0], **n[1])
                else:
                    graph.node(n)
            return graph

        def add_edges(graph, edges): #stolen from http://matthiaseisen.com/articles/graphviz/
            for e in edges:
                if isinstance(e[0], tuple):
                    graph.edge(*e[0], **e[1])
                else:
                    graph.edge(*e)
            return graph

        g = digraph()
        g = add_nodes(g, [(label_to_numberlabel(self.q0), {'color':'green' if self.q0 in self.F else 'black',
                                     'shape': 'hexagon', 'label':'start'})])
        states = list(set(self.Q)-{self.q0})
        g = add_nodes(g, [(label_to_numberlabel(state),{'color': 'green' if state in self.F else 'black',
                                  'label': str(i)})
                          for state,i in zip(states,range(1,len(states)+1))])

        def group_edges():
            def clean_line(line,group):
                line = line.split(separator)
                line = sorted(line) + ["END"]
                in_sequence= False
                last_a = ""
                clean = line[0]
                if line[0] in group:
                    in_sequence = True
                    first_a = line[0]
                    last_a = line[0]
                for a in line[1:]:
                    if in_sequence:
                        if a in group and (ord(a)-ord(last_a))==1: #continue sequence
                            last_a = a
                        else: #break sequence
                            #finish sequence that was
                            if (ord(last_a)-ord(first_a))>1:
                                clean += ("-" + last_a)
                            elif not last_a == first_a:
                                clean += (separator + last_a)
                            #else: last_a==first_a -- nothing to add
                            in_sequence = False
                            #check if there is a new one
                            if a in group:
                                first_a = a
                                last_a = a
                                in_sequence = True
                            if not a=="END":
                                clean += (separator + a)
                    else:
                        if a in group: #start sequence
                            first_a = a
                            last_a = a
                            in_sequence = True
                        if not a=="END":
                            clean += (separator+a)
                return clean


            edges_dict = {}
            for state in self.Q:
                for a in self.alphabet:
                    edge_tuple = (label_to_numberlabel(state),label_to_numberlabel(self.delta[state][a]))
                    # print(str(edge_tuple)+"    "+a)
                    if not edge_tuple in edges_dict:
                        edges_dict[edge_tuple] = a
                    else:
                        edges_dict[edge_tuple] += separator+a
                    # print(str(edge_tuple)+"  =   "+str(edges_dict[edge_tuple]))
            for et in edges_dict:
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_lowercase)
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_uppercase)
                edges_dict[et] = clean_line(edges_dict[et], "0123456789")
                edges_dict[et] = edges_dict[et].replace(separator,",")
            return edges_dict

        edges_dict = group_edges()
        g = add_edges(g,[(e,{'label':edges_dict[e]}) for e in edges_dict])
        # print('\n'.join([str(((str(state),str(self.delta[state][a])),{'label':a})) for a in self.alphabet for state in
        #                  self.Q]))
        # g = add_edges(g,[((label_to_numberlabel(state),label_to_numberlabel(self.delta[state][a])),{'label':a})
        #                  for a in self.alphabet for state in self.Q])
        display(Image(filename=g.render(filename='img/automaton')))

    def minimal_diverging_suffix(self,state1,state2): #gets series of letters showing the two states are different,
        # i.e., from which one state reaches accepting state and the other reaches rejecting state
        # assumes of course that the states are in the automaton and actually not equivalent
        res = None
        # just use BFS til you reach an accepting state
        # after experiments: attempting to use symmetric difference on copies with s1,s2 as the starting state, or even
        # just make and minimise copies of this automaton starting from s1 and s2 before starting the BFS,
        # is slower than this basic BFS, so don't
        seen_states = set()
        new_states = {("",(state1,state2))}
        while len(new_states) > 0:
            prefix,state_pair = new_states.pop()
            s1,s2 = state_pair
            if len([q for q in [s1,s2] if q in self.F])== 1: # intersection of self.F and [s1,s2] is exactly one state,
                # meaning s1 and s2 are classified differently
                res = prefix
                break
            seen_states.add(state_pair)
            for a in self.alphabet:
                next_state_pair = (self.delta[s1][a],self.delta[s2][a])
                next_tuple = (prefix+a,next_state_pair)
                if not next_tuple in new_states and not next_state_pair in seen_states:
                    new_states.add(next_tuple)
        return res