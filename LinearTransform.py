import dynet as dy

class LinearTransform:
    def __init__(self,input_dim,output_dim,pc):
        self.W_param = pc.add_parameters((output_dim, input_dim)) #parameter object
        self.b_param = pc.add_parameters((output_dim)) #parameter object
        self.store_expressions()
        
    def store_expressions(self):
        self.W = dy.parameter(self.W_param) 
        self.bias = dy.parameter(self.b_param)
        
    def apply(self,input_vec_expression):
        return self.W*input_vec_expression + self.bias