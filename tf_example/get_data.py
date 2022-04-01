import os
import shutil

import tensorflow as tf

import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42


def get_data():
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    dataset = tf.keras.utils.get_file('../data/aclImdb_v1.tar.gz', url, untar=True, cache_dir='../data',
                                      cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    # remove_dir = os.path.join(train_dir, 'unsup')
    # shutil.rmtree(remove_dir)
    raw_train_ds = tf.keras.utils.text_dataset_from_directory('../data/aclImdb/train', batch_size=batch_size,
                                                              validation_split=0.2, subset='training', seed=seed)
    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = tf.keras.utils.text_dataset_from_directory('../data/aclImdb/train', batch_size=batch_size,
                                                        validation_split=0.2, subset='validation', seed=seed)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = tf.keras.utils.text_dataset_from_directory('../data/aclImdb/test', batch_size=batch_size)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return class_names, train_ds, val_ds, test_ds
