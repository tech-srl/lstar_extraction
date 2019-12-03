import dynet as dy
from Helper_Functions import map_nested_dict

class GRUCell: 
    def __init__(self,input_dim,output_dim,pc):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pc = pc
        self.gate_names = ["z","r","htilde"] 
        self.gate_activations = [dy.logistic]*2+[dy.tanh]
        #todo: generalise to layers
        self.parameters = {"W":{"x":{},"h":{}},"b":{}}
        for gate in self.gate_names:
            self.parameters["W"]["x"][gate] = self.pc.add_parameters((self.output_dim,self.input_dim))
            self.parameters["W"]["h"][gate] = self.pc.add_parameters((self.output_dim,self.output_dim)) #takes its own previous output
            self.parameters["b"][gate] = self.pc.add_parameters((self.output_dim))
        self.parameters["h0"] = self.pc.add_parameters((self.output_dim))
        self.parameters["h0"].clip_inplace(-1,1)

        self.store_expressions()
        
        
    def store_expressions(self):
        self.expressions = map_nested_dict(self.parameters,dy.parameter)
        self.parameters["h0"].clip_inplace(-1,1)
        self.initial_h = self.parameters["h0"].expr()


                    
    def gate_vecs(self,ht1,xt):
        b = self.expressions["b"]
        W = self.expressions["W"]
        gate_vecs = {}
        for g,activation in zip(self.gate_names,self.gate_activations): 
            hin = ht1 if not g=="htilde" else dy.cmult(gate_vecs["r"],ht1)
            gate_vecs[g] = activation(dy.affine_transform([b[g],
                                                         W["x"][g],xt,
                                                         W["h"][g],ht1]))
        return gate_vecs
    
    def gate_and_next_vecs(self,ht1,xt):
        v = self.gate_vecs(ht1,xt)
        h = dy.cmult(v["z"],ht1)+dy.cmult(1-v["z"],v["htilde"])
        res = v
        res.update({"h":h})
        return res
            
from functools import reduce
from operator import add
class GRUNetworkState:
    def __init__(self,hs=None,full_vec=None,hidden_dim=None):
        if not None in [full_vec,hidden_dim]:
            hvec = full_vec
            self.hs = [dy.inputVector(hvec[i*hidden_dim:(i+1)*hidden_dim]) for i in range(int(len(hvec)/hidden_dim))]
        elif not None in [hs]:            
            self.hs = hs #list of h expressions
        else:
            raise MissingInput()
    
    def output(self):
        return self.hs[-1]

    def as_vec(self):
        return reduce(add,[h.value() for h in self.hs])
        # return np.concatenate([h.npvalue() for h in self.hs]).tolist()

    
class GRUNetwork:
    def __init__(self,num_layers=None,input_dim=None,hidden_dim=None,pc=None,output_dim=None):
        if None in [num_layers,input_dim,hidden_dim,pc] or (num_layers <= 0):
            raise MissingInput()
        if None is output_dim:
            output_dim = hidden_dim
            
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pc = pc
        self.state_class = GRUNetworkState

        self.layers = []
        if self.num_layers > 1:
            self.layers.append(GRUCell(self.input_dim,self.hidden_dim,self.pc))
            for _ in range(num_layers-2):
                self.layers.append(GRUCell(self.hidden_dim,self.hidden_dim,self.pc))
            self.layers.append(GRUCell(self.hidden_dim,self.output_dim,self.pc))
        else:
            self.layers.append(GRUCell(self.input_dim,self.output_dim,self.pc))
            
    def all_gate_and_next_vecs(self,state,input_vec):
        res = []
        x = input_vec
        for layer,h in zip(self.layers,state.hs):
            res.append(layer.gate_and_next_vecs(h,x))
            x = res[-1]["h"] #output of one layer is input to the next
        return res
                       
    def next_state(self,state,input_vec):
        v = self.all_gate_and_next_vecs(state,input_vec)
        hs = [lvals["h"] for lvals in v]
        return GRUNetworkState(hs=hs)
        
    def store_expressions(self):
        for l in self.layers:
            l.store_expressions()
        self.initial_state = GRUNetworkState(hs=[l.initial_h for l in self.layers])