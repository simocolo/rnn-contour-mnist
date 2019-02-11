import os
import numpy as np
import argparse
import math

from scipy import misc
from skimage import color
from skimage import measure
from skimage.draw import ellipse
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action='store', dest='train_path', default='data/train',
                        help='Path to train folder')
    parser.add_argument('--dev_path', action='store', dest='dev_path', default='data/dev',
                        help='Path to dev folder')
    parser.add_argument('--test_path', action='store', dest='test_path', default='data/test',
                        help='Path to test folder')
    opt = parser.parse_args()

    if not os.path.exists(opt.train_path):
        os.makedirs(opt.train_path)
    if not os.path.exists(opt.dev_path):
        os.makedirs(opt.dev_path)
    if not os.path.exists(opt.test_path):
        os.makedirs(opt.test_path)

    train_dataset = datasets.MNIST(root='./data', train=True,
                                download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    train_loader = DataLoader(train_dataset)

    test_dataset = datasets.MNIST(root='./data', train=False,
                                download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    test_loader = DataLoader(test_dataset)

    for i, image in enumerate(train_loader):
        if(i<50000):
            folder = opt.train_path
        elif(i>=50000):
            folder = opt.dev_path
        write_file_line(folder, image)

    for i, image in enumerate(test_loader):
        folder = opt.test_path
        write_file_line(folder, image)

def write_file_line(folder, image):
    img, tgt = image
    img = img.view(28,28).numpy()
    tgt = tgt.numpy()[0]

    gimg = color.colorconv.rgb2grey(img)

    contours = measure.find_contours(gimg, 0.8)
    contours = affine_transform(contours)

    with open(os.path.join(folder,"data.txt"), "a") as f:
        string_contours = get_contours_string(contours)
        f.write(string_contours)
        f.write('\t')
        f.write(str(tgt))
        f.write('\n')

def affine_transform(contours):
    max_x=0
    min_x =math.inf
    max_y=0
    min_y =math.inf
    for n, contour in enumerate(contours):
        max_xy = np.max(contour, axis=0)
        min_xy = np.min(contour, axis=0)
        max_x = max(max_x, max_xy[0])
        max_y = max(max_y, max_xy[1])
        min_x = min(min_x, min_xy[0])
        min_y = min(min_y, min_xy[1])

    sx = 28/(max_x-min_x)
    sy = 28/(max_y-min_y)
    m = (sy if sy<sx else sx)
    tx = -min_x * m
    ty = -min_y * m
    
    for n, contour in enumerate(contours):
        for p in contour:
            p[0] = p[0]*m + tx
            p[1] = p[1]*m + ty

    return contours

def get_contours_string(contours):
    contour_string = ''
    for n, contour in enumerate(contours):
        contour_copy = contour.copy()
        appr_contour = np.rint(approximate_polygon(contour_copy, tolerance=0.8))
        for idc, c in enumerate(appr_contour):
            if idc>0: contour_string += ' '
            contour_string += '{:d}'.format(int(c[0])) + ' ' + '{:d}'.format(int(c[1]))
        contour_string += ' , '
    return contour_string

def img_to_contours_string(file_name):
    fimg = misc.imread(file_name)
    gimg = color.colorconv.rgb2grey(fimg)
    contours = measure.find_contours(gimg, 0.8)
    contours = affine_transform(contours)

    plt.cla()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 0], contour[:, 1], linewidth=2)
        plt.pause(1)
        
    contour_string = get_contours_string(contours)

    return contour_string

if __name__ == '__main__':
    main()