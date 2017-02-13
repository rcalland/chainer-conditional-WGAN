from __future__ import print_function

import collections
import os
import numpy
import chainer
from random import shuffle
from chainer.cuda import cupy

from PIL import Image, ImageOps

class FlexibleImageDataset(chainer.datasets.LabeledImageDataset):

    def __init__(self, paths, root='.', mean=0.0, normalize=True, size=(128,128), dtype=numpy.float32,
                 label_dtype=numpy.int32):

        super(FlexibleImageDataset, self).__init__(paths, root, dtype, label_dtype)
        self.mean = numpy.asarray(mean)
        self.normalize = normalize
        self.size = size

    def get_class_weights(self, gpu_id=-1):
        classes = [x[1] for x in self._pairs]
        collect = collections.Counter(classes)
        weights = [float(item[1]) for item in collect.iteritems()]
        total = sum(weights)
        weights[:] = [ 1.0 - (x / total) for x in weights]
        if gpu_id >= 0:
            return cupy.array(weights, "float32")
        else:
            return numpy.array(weights, "float32")

    def summary(self, freq=False):
        print("Dataset contains {} entries".format(len(self._pairs)))
        classes = [x[1] for x in self._pairs]
        print("Number of classes: {}".format(len(set(classes))))
        if freq:
            print(collections.OrderedDict(collections.Counter(classes)))

    def get_example(self, i, transpose=True):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)

        f = Image.open(full_path).convert("RGB")
        #f = ImageOps.autocontrast(f)
        #f = ImageOps.equalize(f)
        f = f.resize(self.size)#, Image.ANTIALIAS)

        try:
            image = numpy.asarray(f, dtype=self._dtype)
        finally:
            # Only pillow >= 3.0 has 'close' method
            if hasattr(f, 'close'):
                f.close()

        #if image.ndim == 2:
            # image is greyscale
        #    image = image[:, :, numpy.newaxis]
        
        label = numpy.array(int_label, dtype=self._label_dtype)
        #return image.transpose(2, 0, 1), label

        #image = image.astype(numpy.float32)
        image -= self.mean
        #if self.normalize:
        image /= 255.0
        image = image.astype(numpy.float32)

        if transpose:
            return image.transpose(2,0,1), label
        else:
            return image, label