import torch
import librosa
import os
import numpy as np
from torch.autograd import Variable
from sys import stderr

print(torch.__version__)

CONTENT_FILENAME = "/home/naveen/neural-style-audio-tf/inputs/DT10.mp3"
STYLE_FILENAME = "/home/naveen/neural-style-audio-tf/inputs/BO10.mp3"

N_FFT = 2048
def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs

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

conv = torch.nn.Conv2d(N_CHANNELS, N_FILTERS, kernel_size = (1, 11))

content_features_tensor = conv(content_var)
style_features = conv(style_var)

relu = torch.nn.ReLU()

content_features = relu(content_features_tensor).data.numpy()
style_features = relu(style_features).data.numpy()

features = np.reshape(style_features, (-1, N_FILTERS))
style_gram = np.matmul(features.T, features)/N_SAMPLES

ALPHA= 1e-2
learning_rate= 1e-3
iterations = 100
result = None

x = Variable(torch.randn(1, N_CHANNELS, 1, N_SAMPLES).type(torch.FloatTensor)*1e-3, requires_grad = True)
# x = Variable(torch.from_numpy(a_content_tf), requires_grad = True)
conv_x = conv(x)
net_x = relu(conv_x)
loss = torch.nn.MSELoss()
style_loss = 0
target = Variable(torch.FloatTensor(content_features), requires_grad = False)
content_loss = 2 * ALPHA * loss(net_x, target)
size = 1
for val in list(net_x.shape):
	size = size * val
net_x = net_x.resize(int(size/N_FILTERS), N_FILTERS)
gram = np.matmul(net_x.data.numpy().T, net_x.data.numpy())/ N_SAMPLES
style_loss = 2 * np.square(np.subtract(gram, style_gram)).mean()

# style_loss = 2 * loss(gram, style_gram)
loss = content_loss + style_loss
print(loss)


