# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:26:20 2020

@author: lenovo
"""

import numpy as np
import tensorflow as tf


def masked_mse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)

A = np.array([[1.,2.],[0.,3.]])
B = np.array([[1.,2.],[0,2.]])
with tf.Session() as sess:
    pred = tf.constant(A, tf.float32)
    gdt = tf.constant(B, tf.float32)
    loss = masked_mse_tf(pred, gdt,null_val = np.nan)
    ls = sess.run(loss)
    print(ls)