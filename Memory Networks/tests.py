import torch
from torch import optim
import random
from simple_read_write import SimpleMemoryNetwork


def generate_identity_data(input_size, dataset_size, min_sequence_length=5, max_sequence_length=5):
	input_data = []
	output_data = []
	for _ in range(dataset_size):
		sequence_length = random.randint(min_sequence_length, max_sequence_length)
		sequence = torch.rand([sequence_length, input_size]).round()
		input_data.append(sequence)
		output_data.append(sequence)
	return input_data, output_data


# TODO: Supply data generating function.
def test():
	model = SimpleMemoryNetwork(input_size=10, memory_width=20, memory_length=50, output_size=10)
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	input_data, output_data = generate_identity_data(10, 10000)
	loss = 0
	batch_count = 0
	batch_size = 20
	for i in range(len(input_data)):
		batch_count += 1
		outs = []
		for x in input_data[i]:
			outs.append(model(x).view([1,10]))
		loss += torch.sum((torch.cat(outs, dim=0) - output_data[i])**2)
		model.reset()

		if batch_count == batch_size:
			batch_count = 0
			print(loss)
			loss.backward()
			optimizer.step()
			loss = 0

	for i in range(len(input_data)):
		outs = []
		for x in input_data[i]:
			outs.append(model(x).view([1,10]))
		print(output_data[i])
		print(outs)


print(test())

