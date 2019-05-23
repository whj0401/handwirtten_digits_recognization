'''
all data comes from The Mnist Database (http://yann.lecun.com/exdb/mnist/)
'''
import binascii
import numpy as np

mnist_dir = './mnist_database'


def read_image_set(file_path):
    file = open(file_path, 'rb')
    with file:
        magic_num = file.read(4)
        print('magic number: ', binascii.hexlify(magic_num))
        num_images = int(binascii.hexlify(file.read(4)), 16)
        print('number of images: ', num_images)
        num_rows = int(binascii.hexlify(file.read(4)), 16)
        num_columns = int(binascii.hexlify(file.read(4)), 16)
        print('image size: %dx%d' % (num_rows, num_columns))
        images_data = file.read(num_images * num_rows * num_columns)
        images = np.frombuffer(images_data, dtype=np.uint8).reshape(num_images, num_rows, num_columns, 1) # only 1 channel
        file.close()
        return images


def read_label_set(file_path):
    file = open(file_path, 'rb')
    with file:
        magic_num = file.read(4)
        print('magic number: ', binascii.hexlify(magic_num))
        num_items = int(binascii.hexlify(file.read(4)), 16)
        print('number of items: ', num_items)
        labels_data = file.read(num_items)
        labels = np.frombuffer(labels_data, dtype=np.uint8)
        # labels = np.zeros((num_items, 10), dtype=np.uint8)
        # for i in range(num_items):
        #     labels[i][np.uint8(labels_data[i])] = 1
        file.close()
        return labels


def read_train_data():
    images_file_path = mnist_dir + '/train-images.idx3-ubyte'
    labels_file_path = mnist_dir + '/train-labels.idx1-ubyte'
    train_images = read_image_set(images_file_path)
    train_labels = read_label_set(labels_file_path)
    return train_images, train_labels


def read_test_data():
    images_file_path = mnist_dir + '/t10k-images.idx3-ubyte'
    labels_file_path = mnist_dir + '/t10k-labels.idx1-ubyte'
    test_images = read_image_set(images_file_path)
    test_labels = read_label_set(labels_file_path)
    return test_images, test_labels

