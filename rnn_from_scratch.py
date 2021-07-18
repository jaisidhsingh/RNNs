import numpy

"""
A basic implementation of an RNN from scratch, using only numpy.
"""

class RNN():
    def __init__(batch_size, vocab_size, num_hidden):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden = num_hidden

    def init_hidden_state(self):
        return np.zeros((self.batch_size, self.hidden))

    def init_params(self):
        input_size = self.vocab_size
        output_size = self.vocab_size

        def normal(tensor_shape):
            return np.random.normal(scale=0.01, size=tensor_shape)

        Wxh = normal((input_size, self.hidden))
        Whh = norman((self.hidden, self.hidden))
        bh = np.zeros(self.hidden)

        Whq = normal((self.hidden, output_size))
        bq = np.zeros(output_size)
        
        params = [Wxh, Whh, bh, Whq, bq]
        return params
    
    def forward(self, x):
        H = self.init_hidden_state()
        Wxh, Whh, bh, Whq, bq = self.init_params
        outputs = []
        for inp in x:
            H = np.tanh(np.dot(inp, Wxh) + np.dot(H, Whh) + bh)
            Y = np.dot(H, Whq) + bq
            outputs.append(Y)
        return (np.concatenate(outputs, axis=0), H)



