Tensor Train polynomial classifier in MATLAB&copy;/Octave&copy;
----------------------------------------------------------------------------------------------------------------

This package contains MATLAB/Octave implementations for training the Tensor Train polynomial classifier with both the least squares and logistic regression methods.

1. Functions
------------

* [x, res, err]=ttls(a, b, n, r, gamma)

Trains the Tensor Train polynomial classifier with least squares.

* [x, res, err]=ttlr(a, b, n, r, gamma)

Trains the Tensor Train polynomial classifier with logistic regression.

* test_mnist

Trains the Tensor Train polynomial classifier on the MNIST benchmark. 

* test_usps

Trains the Tensor Train polynomial classifier on the USPS benchmark. 


2. Reference
------------

Parallelized Tensor Train Learning of Polynomial Classifiers

https://arxiv.org/abs/1612.06505

Authors: Zhongming Chen, Kim Batselier, Johan A.K. Suykens, Ngai Wong
