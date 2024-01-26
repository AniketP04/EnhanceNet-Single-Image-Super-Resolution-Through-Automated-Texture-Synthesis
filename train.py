import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
import cv2
from model import * 
from dataloader import * 

'''
EnhanceNet Implementation in PyTorch by Erik Quintanilla 

Single Image Super Resolution 

https://arxiv.org/abs/1612.07919/
'''
 
cuda = torch.cuda.is_available()
height = 128
width = 128
channels = 3
lr = .0009 
b1 = .5 
b2 = .9
batch_size = 3
n_epochs= 5
hr_shape = (height, width)

save_interval = 100 

generator = Generator(residual_blocks=10)
discriminator = Discriminator(input_shape=(channels, *hr_shape))
features_2 = Vgg_Features(pool_layer_num = 9) #9 here is the actual index, but it's the second pooling layer 
features_5 = Vgg_Features(pool_layer_num = 36) #36 here is the actual index, but it's the fifth pooling layer 

generator = generator.cuda()
discriminator = discriminator.cuda()
loss = torch.nn.MSELoss().cuda() 
features_2.cuda()
features_5.cuda() 

g_opti = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
d_opti = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

curdir = ""
imagedir = np.load(curdir + "_ids.npy")
lowres = "S:/" + curdir + "_128/"
highres = "S:/" + curdir + "_512/"
gen = Dataset(ids = imagedir, lr = lowres, hr = highres)
train_loader = DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = iter(train_loader)

load_weights = True

if load_weights:
	print("Loading old weights...")
	tmp = torch.load("S:/EnhanceNet-PyTorch/saved_models/generator_0_4800.pth")
	generator.load_state_dict(tmp)
	tmp = torch.load("S:/EnhanceNet-PyTorch/saved_models/discriminator_0_4800.pth")
	discriminator.load_state_dict(tmp)
	print("Best old weights loaded!")

Tensor = torch.cuda.FloatTensor

for epoch in range(n_epochs):
	for i, (lr, hr) in enumerate(train_loader):
	
		lr = Variable(lr.type(Tensor))
		hr = Variable(hr.type(Tensor))
		valid = Variable(Tensor(np.ones((batch_size, *discriminator.output_shape))), requires_grad=False)
		fake = Variable(Tensor(np.zeros((batch_size, *discriminator.output_shape))), requires_grad=False)
		
		'''Generator'''
		
		g_opti.zero_grad() 
		
		generated_hr = generator(lr) 
		
		verdict = discriminator(generated_hr) 
		
		#perceptual loss uses both the second and fifth pooling layer.
		#_2, _5 here denote the pooling layer
		generated_features = features_2(generated_hr)
		real_features = features_2(hr)
		feature_loss_2 = loss(generated_features, real_features.detach())
		
		generated_features = features_5(generated_hr)
		real_features = features_5(hr)
		feature_loss_5 = loss(generated_features, real_features.detach())
		
		total_feature_loss = (.2*feature_loss_2) + feature_loss_5
		
		g_loss = loss(verdict, valid) +  total_feature_loss 
		
		#backpop that loss
		g_loss.backward()
		#update our optimizer 
		g_opti.step()

		'''Discriminator'''

		d_opti.zero_grad() 
		
		hr_imgs = torch.cat([discriminator(hr), discriminator(generated_hr.detach())], dim=0)
		hr_labels = torch.cat([valid, fake], dim=0)
		idxs = list(range(len(hr_labels)))
		idxs = np.random.shuffle(idxs)
		hr_imgs = hr_imgs[idxs] 
		hr_labels = hr_labels[idxs]

		d_loss = loss(hr_imgs, hr_labels)		
		d_loss.backward() 
		d_opti.step()
		
		print("D: %f G: %f \t Epoch: (%i/%i) Batch: (%i/%i)" %(d_loss.item(), g_loss.item(), epoch, n_epochs, i, len(train_loader)))
		if i % save_interval == 0:
			#put the channels back in order!
			generated_hr = generated_hr[:, [2,1,0]]
			hr = hr[:, [2,1,0]]
			#fancy grid so we can view
			generated_hr = make_grid(generated_hr, nrow=1, normalize=True)
			hr = make_grid(hr, nrow=1, normalize=True)
			tmp = torch.cat((hr, generated_hr), -1)
			save_image(tmp, "samples/%d.png" % i, normalize=False)
			#save_image(lr, "samples/originals/%d.png" % i, normalize=False)
			#save generator and discriminator 
			torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
			torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
		


































