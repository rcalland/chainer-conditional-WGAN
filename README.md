# Conditional Wasserstein GAN

Conditional WGAN [forked from this implementation of a WGAN](https://github.com/hvy/chainer-wasserstein-gan) and modified to condition on the input labels [as described in this paper](https://arxiv.org/abs/1411.1784).

## Run

Train the models with CIFAR-10. Images will be randomly sampled from the generator after each epoch, and saved under a subdirectory `result/` (which is created automatically). The user can also specify the output directory as shown below. To resume training from a snapshot, add the `--resume snapshot_epoch_N` argument.

```bash
python train.py --batch-size 64 --epochs 100 --gpu 1 --output "/path/to/output"
```
