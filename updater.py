import numpy as np

from chainer import training, reporter, cuda
from chainer import Variable


class WassersteinGANUpdater(training.StandardUpdater):
    def __init__(self, *, iterator, noise_iterator, optimizer_generator,
                 optimizer_critic, num_classes, device=-1):

        if optimizer_generator.target.name is None:
            optimizer_generator.target.name = 'generator'

        if optimizer_critic.target.name is None:
            optimizer_critic.target.name = 'critic'

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'generator': optimizer_generator,
                      'critic': optimizer_critic}

        super().__init__(iterators, optimizers, device=device)

        if device >= 0:
            cuda.get_device(device).use()
            [optimizer.target.to_gpu() for optimizer in optimizers.values()]

        self.xp = cuda.cupy if device >= 0 else np
        self.num_classes = num_classes

    @property
    def optimizer_generator(self):
        return self._optimizers['generator']

    @property
    def optimizer_critic(self):
        return self._optimizers['critic']

    @property
    def generator(self):
        return self._optimizers['generator'].target

    @property
    def critic(self):
        return self._optimizers['critic'].target

    @property
    def x(self):
        return self._iterators['main']

    @property
    def z(self):
        return self._iterators['z']

    def onehot(self, notonehot):
        #num_classes = 10
        onehot = self.xp.eye(self.num_classes)[notonehot].astype(self.xp.float32)
        return onehot

    def next_batch(self, iterator):
        batch = self.converter(iterator.next(), self.device)

        if len(batch) is 2:
            batch_img = batch[0]
            batch_lbl = batch[1]
            _onehot = self.onehot(batch_lbl)

            return Variable(batch_img), _onehot
        else:
            return Variable(batch)

    def sample(self):

        """Return a sample batch of images."""

        z = self.next_batch(self.z)
        #l = self.xp.random.randint(43, size=self.z.batch_size)
        l = self.xp.ones((self.z.batch_size)).astype(np.uint8)
        l_1h = self.onehot(l)
        #print(l_1h.shape)
        x = self.generator(z, l_1h, test=True)

        # [-1, 1] -> [0, 1]
        x += 1.0
        x /= 2

        return x

    def update_core(self):

        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        # Update critic 5 times
        for _ in range(5):
            # Clamp critic parameters
            self.critic.clamp()

            # Real images
            x_real, x_label = self.next_batch(self.x)
            #x_real -= 1.0
            y_real = self.critic(x_real, x_label)
            y_real.grad = self.xp.ones_like(y_real.data)
            _update(self.optimizer_critic, y_real)

            # Fake images
            z = self.next_batch(self.z)
            x_fake = self.generator(z, x_label)
            y_fake = self.critic(x_fake, x_label)
            y_fake.grad = -1 * self.xp.ones_like(y_fake.data)
            _update(self.optimizer_critic, y_fake)

            reporter.report({
                'critic/loss/real': y_real,
                'critic/loss/fake': y_fake,
                'critic/loss': y_real - y_fake
            })

        # Update generator 1 time
        z = self.next_batch(self.z)
        # random label
        l = self.xp.random.randint(self.num_classes, size=self.z.batch_size)
        l_1h = self.onehot(l)

        x_fake = self.generator(z, l_1h)
        y_fake = self.critic(x_fake, l_1h)
        y_fake.grad = self.xp.ones_like(y_fake.data)
        _update(self.optimizer_generator, y_fake)

        reporter.report({'generator/loss': y_fake})
