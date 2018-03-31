import torch
from torch import nn
import librosa
import os
import numpy as np
from torch.autograd import Variable
from sys import stderr

CONTENT_FILENAME = "/home/naveen/neural-style-audio-tf/inputs/gajiniLyrical.mp3"
STYLE_FILENAME = "/home/naveen/neural-style-audio-tf/inputs/gajiniInstrumental.mp3"

N_FFT = 2048


def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs

class GramMatrix(nn.Module):

	def forward(self, input):
		features = input.view(-1, N_FILTERS)
		G = torch.mm(features.transpose(0,1), features)
		return G.div(N_SAMPLES)

class StyleLoss(nn.Module):

	def __init__(self, target, weight):
		super(StyleLoss, self).__init__()
		self.target = target.detach() * weight
		self.weight = weight
		self.gram = GramMatrix()
		self.criterion = nn.MSELoss()
	
	def forward(self, input):
		self.output = input.clone()
		self.G = self.gram(input)
		self.G.mul_(self.weight)
		self.loss = self.criterion(self.G, self.target)
		return self.output

	def backward(self, retain_graph=True):
		self.loss.backward(retain_graph=retain_graph)
		return self.loss


a_content, fs = read_audio_spectum(CONTENT_FILENAME)
a_style, fs = read_audio_spectum(STYLE_FILENAME)

N_SAMPLES = a_content.shape[1]
N_CHANNELS = a_content.shape[0]
a_style = a_style[:N_CHANNELS, :N_SAMPLES]

N_FILTERS = 4096

a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

a_content_tf = np.transpose(a_content_tf, (0, 3, 1, 2))
a_style_tf = np.transpose(a_style_tf, (0, 3, 1, 2))

std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)*std

content_var = Variable(torch.from_numpy(a_content_tf), requires_grad = False)
style_var = Variable(torch.from_numpy(a_style_tf), requires_grad = False)

model = torch.nn.Sequential(
			torch.nn.Conv2d(N_CHANNELS, N_FILTERS, kernel_size = (1, 11)),
			torch.nn.ReLU()
		)

content_features = model(content_var)
style_features = model(style_var)

style_features = style_features.view(-1, N_FILTERS)
# style_gram = torch.mm(style_features.transpose(0,1), style_features)/N_SAMPLES

ALPHA = 1e-2
learning_rate = 1e-2
iterations = 100
result = None

# x = Variable(torch.randn(1, N_CHANNELS, 1, N_SAMPLES).type(torch.FloatTensor)*1e-3, requires_grad = True)
x = torch.nn.Parameter(torch.from_numpy(a_content_tf), requires_grad = True)

loss_fn = torch.nn.MSELoss()
style_loss = 0

optimizer = torch.optim.LBFGS([x], lr=learning_rate)
gram = GramMatrix()
style_gram_target = gram(style_features)
# style_gram_target = Variable(style_gram.data, requires_grad = False)


# def closure_version1():
# 	global x
# 	optimizer.zero_grad()
# 	output = model(x)
# 	net_x = output.view(-1, N_FILTERS)
# 	gram = Variable((torch.mm(net_x.transpose(0,1), net_x).data/ N_SAMPLES), requires_grad = True)
# 	loss = StyleLoss(gram, style_gram_target)
# 	print(t, loss.data[0])
# 	gr = torch.autograd.grad([loss], [x])
# 	x = x + learning_rate * gr[0]
# 	loss.backward()
# 	return loss

gram = GramMatrix()
style_loss_module = StyleLoss(style_gram_target, 1000)
model.add_module("style_loss", style_loss_module)

for t in range(10):
	print(t)
	def closure_version2():
		style_score = 0
		optimizer.zero_grad()
		# net_x = model(x)
		# net_x = net_x.view(-1, N_FILTERS)
		# gram = Variable((torch.mm(net_x.transpose(0,1), net_x).data/ N_SAMPLES), requires_grad = True)
		model(x)
		# print(t, loss.data[0])
		style_score += style_loss_module.backward()
		return style_score
	optimizer.step(closure_version2)

x = x.data.numpy()
x = np.transpose(x, (0, 2, 3, 1))
x = np.reshape(x, (-1, N_CHANNELS))

if np.array_equal(x.T, a_content):
	print("equal")
else:
	print("not equal")

print("content = ", a_content)
print("x = ", x)
diff = a_content - x.T
print("diff = ", diff)

# a = np.zeros_like(a_content)
# a[:N_CHANNELS,:] = np.exp(x.T) - 1
# p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
# for i in range(500):
#     S = a * np.exp(1j*p)
#     x = librosa.istft(S)
#     p = np.angle(librosa.stft(x, N_FFT))

# OUTPUT_FILENAME = 'outputs/out.wav'
# librosa.output.write_wav(OUTPUT_FILENAME, x, fs)
# print(OUTPUT_FILENAME)
# print("done")

