import numpy

"""
A basic implementation of an RNN from scratch, using only numpy.
"""

def initialize_params(vocab_size, hidden):
    input_size = vocab_size 
    output_size = vocab_size

    def normal(tensor_shape):
        return np.random.normal(scale=0.01, size=tensor_shape)

    Wxh = normal((input_size, hidden))
    Whh = norman((hidden, hidden))
    bh = np.zeros(hidden)

    Whq = normal((hidden, output_size))
    bq = np.zeros(output_size)
    
    params = [Wxh, Whh, bh, Whq, bq]
    return params

def initialize_rnn_state(batch_size, hidden):
    return np.zeros((batch_size, hidden))

def forward_rnn(inputs, state, params):
    # input tensor shape: (num_time_steps, batch_size, vocab_size)
    Wxh, Whh, bh, Whq, bq = params
    H = state
    outputs = []
    for X in inputs:
        H = np.tanh(np.dot(X, Wxh) + np.dot(H, Whh) + bh)
        Y = np.dot(H, Wqh) + bq
        outputs.append(Y)
    return (np.concatenate(outputs, axis=0), H)


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



