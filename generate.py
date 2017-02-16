from __future__ import print_function

import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plt

import numpy
import math
import argparse

import chainer
from chainer import cuda, serializers, Variable
import chainer.functions as F
import chainer.links as L

from models import Generator
from iterators import RandomNoiseIterator, GaussianNoiseGenerator
from chainer.training import updater, extensions, StandardUpdater

from PIL import Image

from extensions import save_ims

def onehot(notonehot, num_classes):
	onehot = cuda.cupy.eye(num_classes)[notonehot].astype(cuda.cupy.float32)
	return onehot.reshape((1, num_classes))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", "-m", type=str, required=True)
	parser.add_argument("--label", "-l", type=int, default=5)
	parser.add_argument("--gpu", "-g", type=int, default=-1)
	parser.add_argument("--output", "-o", type=str, default="sample.png")
	args = parser.parse_args()

	if args.gpu >= 0:
		chainer.cuda.get_device(args.gpu).use() # use this GPU
	
	generator = Generator()
	serializers.load_npz(args.model, generator)
	generator.to_gpu()

	num_classes = 43
	batchsize = 64

	class_onehot = onehot(args.label, num_classes)
	class_onehot = cuda.cupy.array(class_onehot)
	class_onehot = cuda.cupy.stack([class_onehot] * batchsize, 0)
	class_onehot = class_onehot.reshape(batchsize, num_classes)

	z_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, 100), batchsize)
	batch = cuda.cupy.array(z_iter.next()) 

	x = generator(batch, class_onehot, test=True)
	x += 1
	x /= 2.0

	if cuda.get_array_module(x) == cuda.cupy:
		x = cuda.to_cpu(x.data)
	else:
		x = x.data
	
	save_ims(args.output, x)
	print("- Saved to {}.".format(args.output))


if __name__=="__main__":
	main()