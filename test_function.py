import pandas as pd
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
# import idx2numpy
#
# =np.load('test_train/kmnist-train-imgs.npz')
# np.load('test_train/kmnist-train-labels.npz')

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse a group registeration table in the CSV format. ')
    parser.add_argument('--csv', help='CSV filename')
    # arg.[option]
    args = parser.parse_args()
    print('filenmae : ' + args.csv)
    # f = gzip.open(os.path.join(os.path.dirname(__file__),'test_train/train-images-idx3-ubyte.gz'),'r')
    # image_size = 28
    # num_images = 5
    # f.read(16)
    # buf = f.read(image_size * image_size * num_images)
    # data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    # data = data.reshape(num_images, image_size, image_size, 1)
    # print(data[0])
    # print(type(data[0]))
    # print(len(data[0]))
    # image = np.asarray(data[0]).squeeze()
    # print(len(image))
    # plt.imshow(image)
    # plt.show()

    # f = gzip.open(os.path.join(os.path.dirname(__file__),'test_train/train-labels-idx1-ubyte.gz'),'r')
    # f.read(8)
    # for i in range(0,50):
    #     buf = f.read(1)
    #     labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    #     print(labels)



#     file = os.path.join(os.path.dirname(__file__),'test_train/train-labels-idx1-ubyte.gz','r')
#     arr = idx2numpy.convert_from_file(file)