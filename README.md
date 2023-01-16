# Differentiable Discrete Samplers (d2sample)

`d2sample` is a collection of PyTorch differentiable samplers for discrete objects, with associated examples and layers (TBD). Install by first cloning recursively:

```
git clone --recursive git@github.com:sscardapane/d2sample.git
```

Then (for now) add to the path:

```
import sys
sys.path.append('./d2sample/')
```

## Requirements

* PyTorch == 1.13.1
* [functorch](https://pytorch.org/functorch/stable/)

## Implemented algorithms

### **$k$-subset sampling** (see [notebook](./notebooks/1_SubsetSampling.ipynb)):

1. Gumbel-Softmax with continuous top-$k$ relaxation ([Xie \& Ermon, 2019](https://arxiv.org/abs/1901.10517)). For $k=1$ this reduces to the standard Gumbel-Softmax reparameterization available inside PyTorch.
2. Top-k selection with I-MLE ([Niepert, Minervini, \& Franceschi, 2021](https://arxiv.org/abs/2106.01798)).
3. SIMPLE: Subset Implicit Likelihood Estimation ([Ahmed, Zeng, Niepert, \& Van den Broeck, 2022](https://arxiv.org/abs/2210.01941)).