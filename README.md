# rnn-contour-mnist
A recurrent neural network applied to contours from MNIST digits in pytorch.

This project is used to train a RNN (a sequence to sequence model) on contours extracted from MNIST images.

It uses seq2seq library from IBM - source https://github.com/IBM/pytorch-seq2seq

Contours are taken from images using skimage's class 'measure'. Then they are written into three separate text files for training, developing and testing.

Each line in these files represents the contours and each contour is separated by a comma. Each point of a contour is approx to a couple of integers representing x and y values (coordinates of that point in the image).
Separated by a TAB at the end of the line we write the corresponding digit (the library is used improperly cause we don't need a seq2seq to classify a digit but this library actually works so well..).

So for example the sequence below contains x and y values of two contours representing a '9':

    28 3 27 1 13 9 14 5 13 2 11 0 8 0 3 5 0 11 0 14 4 19 6 19 28 3 , 9 10 6 13 3 13 6 6 9 3 9 10 , 	9



## Create the datasets

In order to create the three datasets just call 

    write_contour_file.py --train_path 'data/train' --dev_path 'data/dev' --test_path 'data/test'

MNIST database will be then downloaded and the program iterates over it to create 50000 lines for training, 5000 for developing and 5000 for testing.

## Training the model

To train the model call:

    train.py --train_path 'data/train' --dev_path 'data/dev' --test_path 'data/test'

    2019-02-11 15:47:34,856 seq2seq.trainer.supervised_trainer INFO     Progress: 0%, Train Perplexity: 13.9967
    2019-02-11 15:47:36,609 seq2seq.trainer.supervised_trainer INFO     Progress: 0%, Train Perplexity: 3.6925
    ...
    Dev Perplexity: 1.0562, Accuracy: 0.9868
    Test Perplexity: 1.0378, Accuracy: 0.9897

## Load a model

To a load a pre-trained model call:

    train.py --load_checkpoint '2019_02_08_11_41_29' --train_path 'data/train' --dev_path 'data/dev' --test_path 'data/test'

Once the program ends the training or loading the saved model, you will able to test results drawing a number by yourself.