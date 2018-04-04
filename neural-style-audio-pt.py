import torch
from torch import nn
import librosa
import os
from torch import optim
import numpy as np
from torch.autograd import Variable
from sys import stderr

INPUT_DIR = '/home/naveen/neural-style-audio-tf/inputs/monochannel/'
CONTENT_FILENAME = 'BO10.mp3'
STYLE_FILENAME = 'DT10.mp3'

CONTENT_WEIGHT = 10
STYLE_WEIGHT = 600
ITERATIONS = 10

N_FFT = 2048

# ALPHA = 1e-2
learning_rate = 1e-2

# use_cuda = torch.cuda.is_available()
use_cuda = False

def get_input_param_optimizer(input_val):
    # this line to show that input is a parameter that requires a gradient
    if use_cuda:
        input_param = nn.Parameter(input_val.clone().data)
    else:
        input_param = nn.Parameter(input_val.data)
    optimizer = optim.LBFGS([input_param], lr = learning_rate, max_iter=10)
    return input_param, optimizer

def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    return x, fs

class GramMatrix(nn.Module):

	def forward(self, input):
		features = input.view(-1, N_FILTERS)
		G = torch.mm(features.transpose(0,1), features)
		return G.div(N_SAMPLES)

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

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


a_content, fs = read_audio_spectum(INPUT_DIR + CONTENT_FILENAME)
a_style, fs = read_audio_spectum(INPUT_DIR + STYLE_FILENAME)

# 10 sec audio requires a lot of memory for processing so taking the first half of audio
N_SAMPLES = int(a_content.shape[0]/5)

N_CHANNELS = 1

a_style = a_style[:N_SAMPLES]
a_content = a_content[:N_SAMPLES]

N_FILTERS = 4096

# # changing x changes the a_content for some reason, so using deepcopy
content_copy = np.copy(a_content)

a_content_tf = np.ascontiguousarray(content_copy.T[None,None,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:])

if use_cuda:
    content_var = Variable(torch.from_numpy(a_content_tf).cuda(), requires_grad = False)
    style_var = Variable(torch.from_numpy(a_style_tf).cuda(), requires_grad = False)
else:
    content_var = Variable(torch.from_numpy(a_content_tf), requires_grad = False)
    style_var = Variable(torch.from_numpy(a_style_tf), requires_grad = False)

model = torch.nn.Sequential(
			torch.nn.Conv1d(N_CHANNELS, N_FILTERS, kernel_size = 4),
			torch.nn.ReLU()
		)

gram = GramMatrix()

if use_cuda:
    gram = gram.cuda()
    model = model.cuda()

content_features = model(content_var)
style_features = model(style_var).clone()

style_features = style_features.view(-1, N_FILTERS)
style_gram_target = gram(style_features)

# # rand_input = Variable(torch.randn(1, N_CHANNELS, 1, N_SAMPLES).type(torch.FloatTensor)*1e-3, requires_grad = True)
# # # x, optimizer = get_input_param_optimizer(rand_input)

x, optimizer = get_input_param_optimizer(content_var)

style_loss_module = StyleLoss(style_gram_target, STYLE_WEIGHT)
# # content_loss_module = ContentLoss(content_features.clone(), CONTENT_WEIGHT)
model.add_module("style_loss", style_loss_module)
# # model.add_module("content_loss", content_loss_module)

t = [0]
while t[0] < ITERATIONS:
	def closure():
		score = 0
		optimizer.zero_grad()
		model(x)
		t[0] += 1
		# total_loss = style_loss_module.backward() + content_loss_module.backward()
		total_loss = style_loss_module.backward()
		score += total_loss
		print(t[0], score.data[0])
		return score
	optimizer.step(closure)

x = x.data.numpy()
x = x.flatten()
print(x.shape)
librosa.output.write_wav('outputs/1dconvo/BODT-1dConv.wav', x, fs)

print("done")

