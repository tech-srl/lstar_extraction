import numpy as np
from Helper_Functions import mean
from sklearn import svm
from copy import deepcopy

class SVMDecisionTreeNode:
    def __init__(self,id):
        self.id = id
        self.has_children = False

    def get_node(self,vector):
        if self.has_children:
            return self._choose_child(vector).get_node(vector)
        return self

    def _choose_child(self,vector):
        if self.is_dim_split:
            return self._dim_choose_child(vector)
        childnum = self.clf.predict([vector]).tolist()[0]
        if childnum == 0:
            return self.zero_child
        return self.one_child

    def _dim_choose_child(self,vector):
        if vector[self.split_dim] > self.split_val:
            return self.high
        return self.low

    def _dim_split_aux(self, split_tuples, split_vals, new_id, split_depth):
        if split_depth == 0 or len(split_tuples)==0:
            return new_id

        split_tuples = deepcopy(split_tuples) #else they're all popping from the same place and run out of dimensions to split, not to mention split asymmetrically!
        margin, self.split_dim = split_tuples.pop(0)

        self.split_val = split_vals[self.split_dim]

        self.high = SVMDecisionTreeNode(self.id)
        self.low = SVMDecisionTreeNode(new_id)
        new_id += 1 # next node will need to have the next id now that low has taken the one we had
        self.has_children = True
        self.is_dim_split = True

        new_id = self.high._dim_split_aux(split_tuples,split_vals,new_id,split_depth-1)
        new_id = self.low._dim_split_aux(split_tuples,split_vals,new_id,split_depth-1)
        return new_id

    # reminder: this function (dim_split) and the next (split) are called only by the overall SVMDecisionTree class,
    # and won't be called on internal nodes of the tree (only on leaves, which represent actual clusters in the partitioning)
    def dim_split(self,agreeing_continuous_visitors,conflicted_continuous_visitor,new_id,split_depth):
        # print("making initial split of depth " + str(split_depth))
        mean_agreeing_vector = []
        for i in range(len(conflicted_continuous_visitor)):
            mean_agreeing_vector.append(mean([visitor[i] for visitor in agreeing_continuous_visitors]))

        margins = [abs(m-v) for m,v in zip(mean_agreeing_vector,conflicted_continuous_visitor)]
        numbered_margins_by_largest = sorted([(margin,i) for i,margin in enumerate(margins)],reverse=True)
        split_vals = [(a+b)/2.0 for a,b in zip(mean_agreeing_vector,conflicted_continuous_visitor)]

        return self._dim_split_aux(numbered_margins_by_largest,split_vals,new_id,split_depth)

    def split(self,agreeing_continuous_visitors,conflicted_continuous_visitor,new_id):
        # print("trying regular svm split")
        x = agreeing_continuous_visitors + [conflicted_continuous_visitor]
        y = [0]*len(agreeing_continuous_visitors) + [1]
        self.clf = svm.SVC(C=10000)
        self.clf.fit(x,y)
        # print("clf used this many support vectors: " + str(self.clf.n_support_))
        self.zero_child = SVMDecisionTreeNode(self.id)
        self.one_child = SVMDecisionTreeNode(new_id)
        new_id += 1
        self.has_children = True
        self.is_dim_split = False
        if not self.clf.predict(x).tolist() == y:
            print("svm classifier failed to obtain perfect split :(")
        return new_id


class SVMDecisionTreeQuantisation: 
    def __init__(self,num_dims_initial_split):
        self.num_dims_initial_split = num_dims_initial_split
        self.top_id = 1 #1-index so it's also a neat count of how many id's we have in general
        self.head = SVMDecisionTreeNode(self.top_id)
        self.had_initial_refine = False
        self.initiated_with_all_rnn_states_to_some_depth = False
        
        self.refinement_doesnt_hurt_other_clusters = True 
        # this is a trait of Decision Tree refinements: they affect only the cluster being refined, 
        # all the rest remain exactly the same (as opposed to, for instance, splitting a dimension 
        # across the board). If you wish to implement a different quantisation, think about whether 
        # yours satisfies this quality and fill this field appropriately)

        pass

    def _get_node(self,vector):
        if not self.had_initial_refine:
            return self.head
        if self.initiated_with_all_rnn_states_to_some_depth:
            return self.nodes[self.clf.predict([vector])[0]].get_node(vector)
        return self.head.get_node(vector)

    def get_partition(self, vector):
        return self._get_node(vector).id

    def refine(self,agreeing_continuous_visitors,conflicted_continuous_visitor):
        # print("refining, H size is " + str(len(agreeing_continuous_visitors)))
        relevant_node = self._get_node(conflicted_continuous_visitor)
        next_id = relevant_node.split(agreeing_continuous_visitors,conflicted_continuous_visitor,self.top_id+1) if \
            self.had_initial_refine else relevant_node.dim_split(agreeing_continuous_visitors,
                                                                 conflicted_continuous_visitor,
                                                                 self.top_id+1,
                                                                 self.num_dims_initial_split)
        self.refined_something = True
        # print("refining - added "+str((next_id-1)-self.top_id)+" states")
        self.top_id = next_id-1
        self.had_initial_refine = True
