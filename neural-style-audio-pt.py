import torch
from torch import nn
import librosa
import os
from torch import optim
import numpy as np
from torch.autograd import Variable
from sys import stderr

CONTENT_FILENAME = "/home/naveen/neural-style-audio-tf/inputs/gettysburg.mp3"
STYLE_FILENAME = "/home/naveen/neural-style-audio-tf/inputs/eminem.mp3"

N_FFT = 2048

def get_input_param_optimizer(input_val):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_val.clone().data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

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


a_content, fs = read_audio_spectum(CONTENT_FILENAME)
a_style, fs = read_audio_spectum(STYLE_FILENAME)

N_SAMPLES = a_content.shape[1]
N_CHANNELS = a_content.shape[0]
a_style = a_style[:N_CHANNELS, :N_SAMPLES]

N_FILTERS = 4096

# changing x changes the a_content for some reason, so using deepcopy
content_copy = np.copy(a_content)

a_content_tf = np.ascontiguousarray(content_copy.T[None,None,:,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

#rearrange axis to meet conv2d input format requirement
a_content_tf = np.transpose(a_content_tf, (0, 3, 1, 2))
a_style_tf = np.transpose(a_style_tf, (0, 3, 1, 2))

# std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
# kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)*std

content_var = Variable(torch.from_numpy(a_content_tf), requires_grad = False)
style_var = Variable(torch.from_numpy(a_style_tf), requires_grad = False)

model = torch.nn.Sequential(
			torch.nn.Conv2d(N_CHANNELS, N_FILTERS, kernel_size = (1, 11)),
			torch.nn.ReLU()
		)

content_features = model(content_var)
style_features = model(style_var).clone()

style_features = style_features.view(-1, N_FILTERS)
gram = GramMatrix()
style_gram_target = gram(style_features)

ALPHA = 1e-2
learning_rate = 1e-2
iterations = 100
result = None

# x = Variable(torch.randn(1, N_CHANNELS, 1, N_SAMPLES).type(torch.FloatTensor)*1e-3, requires_grad = True)
rand_input = Variable(torch.randn(1, N_CHANNELS, 1, N_SAMPLES).type(torch.FloatTensor)*1e-3, requires_grad = True)
# x, optimizer = get_input_param_optimizer(rand_input)
x, optimizer = get_input_param_optimizer(content_var)

gram = GramMatrix()
style_loss_module = StyleLoss(style_gram_target, 300)
content_loss_module = ContentLoss(content_features.clone(), 10)
model.add_module("style_loss", style_loss_module)
# model.add_module("content_loss", content_loss_module)


t = [0]
while t[0] < 100:
	def closure_version2():
		score = 0
		optimizer.zero_grad()
		model(x)
		t[0] += 1
		# total_loss = style_loss_module.backward() + content_loss_module.backward()
		total_loss = style_loss_module.backward()
		score += total_loss
		print(t[0], score.data[0])
		return score
	optimizer.step(closure_version2)

x = x.data.numpy()

#undo the axis remarrangement done before
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

a = np.zeros_like(a_content)
a[:N_CHANNELS,:] = np.exp(x.T) - 1
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, N_FFT))

OUTPUT_FILENAME = 'outputs/eminem_gettysberg.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)
print(OUTPUT_FILENAME)
print("done")

