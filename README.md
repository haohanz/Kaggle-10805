# Rotational Dropout - Image Classifiers Across Geographic Distributions
Awesome Kaggle Competition
https://www.kaggle.com/c/inclusive-images-challenge

The combination of two most powerful techniques in neural network, Dropout and Batch Normalization (BN) often leads to a worse performance. Because Dropout would shift the variance of a specific neural unit when we transfer the state of the network from train to test. Whereas BN would maintain its statistical variance, which is accumulated from the entire learning procedure during the inference phase. We propose Rotational Dropout, which can integrate the merit from both Dropout and Batch Normalization for each building block by keeping consistent variance. Rotational Dropout efficiently mitigates models overfitting problem, remarkably enhances a Convolutional Neural Network's modeling ability on one domain as well as its generalization capacity on another domain without fine-tuning. Moreover, it can be wrapped into many advanced deep networks (ResNet, DenseNet, SENet, etc) to improve their performances without increasing much computational cost as a domain adaptation method.


To avoid the variance shift risks, we propose a new generalization method which keeps the direction information of feature vectors, named Rotational Dropout. Rotational Dropout can avoid variance shift risk appearing in the original Dropout. It fundamentally overcomes the shortcoming of the original Dropout method and can be combined with Batch Normalization to bring significant improvements on state-of-art CNN architectures by powerfully mitigating overfitting problem.

Besides, we conduct extensive statistical experiments to check the performance of Rotational Dropout. Rotational Dropout can remarkably enhance a Convolutional Neural Network's modeling ability on one domain as well as its generalization capacity on another domain without fine-tuning. Moreover, it can be wrapped into many advanced deep networks to improve their performances without increasing much computational cost as a domain adaptation method. We reach 25.1\% of F2 score on the OpenImage dataset by using Rotational Dropout and Batch Normalization on ResNet50. We achieve an error rate of 4.56\% on CIFAR10 and 21.74\% on CIFAR100 for DenseNet, error rate of 4.74\% and 23.23\% for ResNet50 on CIFAR10 and CIFAR100 respectively.

Most importantly, Rotational Dropout can not only be implemented on image classification tasks, but it can also be applied to natural language processing, speech recognition, etc. Our design is suitable for all CNN based architectures, it can be easily installed into every state-of-art CNNs, meanwhile, bring about significant improvement of performance.




Authors: @[Kai Hoo](https://github.com/KaiHoo); @[Haohan Zhang](https://github.com/haohanz)
