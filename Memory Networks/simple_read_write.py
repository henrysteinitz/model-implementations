import torch
from torch.autograd import Variable
from torch.nn import Module, Parameter
from torch.nn.functional import softmax, sigmoid

# Location-addressable read-write memory with feedforward controllers. When memory_width = 1, 
# this reduces to a vanilla recurrent network.
class SimpleMemoryNetwork(Module):
	# 	Parameters:
	#		input_size: Length of each input vector
	# 		memory_width: The length of each hidden state vectors stored in the memory metrix
	#		memory_length: The total number of hidden state vectors stored in the memory matrix
	#		output_size: Length of each output vector
	def __init__(self, input_size, memory_width, memory_length, output_size):
		super().__init__()

		self._memory_width = memory_width
		self._memory_length = memory_length
		scale_factor = .1
		self._output_weights = Parameter(
			scale_factor * torch.randn([output_size, memory_width + input_size]),
			requires_grad=True
		)
		self._output_biases = Parameter(
			scale_factor * torch.randn([output_size]), 
			requires_grad=True
		)
		self._read_key_weights = Parameter(
			scale_factor * torch.randn([memory_length, memory_width + input_size]),
			requires_grad=True
		)
		self._read_key_biases = Parameter(
			scale_factor * torch.randn([memory_length]),
			requires_grad=True
		)
		self._write_key_weights = Parameter(
			scale_factor * torch.randn([memory_length, memory_width + input_size]), 
			requires_grad=True
		)
		self._write_key_biases = Parameter(
			scale_factor * torch.randn([memory_length]), 
			requires_grad=True
		)
		self._write_vector_weights = Parameter(
			scale_factor * torch.randn([memory_width, memory_width + input_size]), 
			requires_grad=True
		)
		self._write_vector_biases = Parameter(
			scale_factor * torch.randn([memory_width]),
			requires_grad=True
		)
		self._memory = Variable(scale_factor * torch.randn([memory_length, memory_width]))
		self._prev_read = Variable(scale_factor * torch.randn([memory_width]))

	def forward(self, x):
		total_input = torch.cat([x, self._prev_read])
		out = sigmoid(torch.mv(self._output_weights, total_input) + self._output_biases)
		read_key = softmax(torch.mv(self._read_key_weights, total_input) + self._read_key_biases)
		write_key = softmax(torch.mv(self._write_key_weights, total_input) + self._write_key_biases)
		write_vector = softmax(torch.mv(self._write_vector_weights, total_input) + self._write_vector_biases)
		self._prev_read = torch.sum(self._memory * read_key.view([self._memory_length, 1]), dim=0)
		self._memory = self._memory * (1 - write_key).view([self._memory_length, 1]) + \
				(write_key.view([self._memory_length, 1]) * write_vector.view([1, self._memory_width])) 
		return out

	def reset(self):
		scale_factor = .1
		self._memory = Variable(scale_factor * torch.randn([self._memory_length, self._memory_width]))
		self._prev_read = Variable(scale_factor * torch.randn([self._memory_width]))

