# rnn-contour-mnist
A recurrent neural network applied to contours from MNIST digits in pytorch.

This project is used to train a RNN (a sequence to sequence model) on contours extracted from MNIST images.

Contours are taken from images using skimage's class 'measure'. Then they are written into three separate text files for training, developing and testing.

Each line in these files represents the contours and each contour is separated by a comma. Each point of a contour is approx to a couple of integers representing x and y values (coordinates of that point in the image).

This project uses seq2seq library from IBM - source https://github.com/IBM/pytorch-seq2seq
