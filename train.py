import argparse

from chainer import datasets, training, iterators, optimizers, optimizer, serializers
from chainer.training import updater, extensions

from models import Generator, Critic
from updater import WassersteinGANUpdater
from extensions import GeneratorSample
from iterators import RandomNoiseIterator, GaussianNoiseGenerator

from data import FlexibleImageDataset
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--resume', '-r', type=str, default=None)
    return parser.parse_args()


def train(args):
    nz = args.nz
    batch_size = args.batch_size
    epochs = args.epochs
    gpu = args.gpu

    # CIFAR-10 images in range [-1, 1] (tanh generator outputs)
    train, _ = datasets.get_cifar10(withlabel=True, ndim=3, scale=2)
    #train._datasets[0] -= 1.0
    #print(train._datasets[1].shape)

    #train, _ = datasets.get_mnist(withlabel=True, ndim=3, scale=2)
    #print(train.shape)

    """
    training_data = FlexibleImageDataset("/mnt/sakuradata2/calland/software/chainer-GTSRB/annotations/GTSRB_test.txt",
                                        size=(32,32))
    train = np.array([training_data.get_example(x)[0] for x in range(len(training_data._pairs))])
    train -= 1.0
    train *= 2
    """

    train_iter = iterators.SerialIterator(train, batch_size)

    z_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, args.nz),
                                 batch_size)

    optimizer_generator = optimizers.RMSprop(lr=0.00005)
    optimizer_critic = optimizers.RMSprop(lr=0.00005)
    generator = Generator()
    optimizer_generator.setup(generator)
    optimizer_critic.setup(Critic())

    updater = WassersteinGANUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_critic=optimizer_critic,
        device=gpu)

    trainer = training.Trainer(updater, stop_trigger=(epochs, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(GeneratorSample(), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'critic/loss',
            'critic/loss/real', 'critic/loss/fake', 'generator/loss']))
   # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
    generator, 'model_epoch_{.updater.epoch}'), trigger=(1, 'epoch'))

    if args.resume:
        # Resume from a snapshot
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    args = parse_args()
    train(args)
