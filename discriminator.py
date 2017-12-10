#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:43:44 2017

@author: no1
"""
import tensorflow as tf
import tools
def create_discriminator(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []
    ndf=64
    '''
    [<tf.Tensor  shape=(1, 16, 16, 64) dtype=float32>,
    <tf.Tensor  shape=(1, 8, 8, 128) dtype=float32>,
    <tf.Tensor  shape=(1, 4, 4, 256) dtype=float32>,
    <tf.Tensor  shape=(1, 3, 3, 512) dtype=float32>,
    <tf.Tensor  shape=(1, 2, 2, 1) dtype=float32>]
    '''
    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 32, 32, in_channels * 2] => [batch, 16, 16, ndf]
    with tf.variable_scope("layer_1"):
        convolved = tools.conv(input, ndf, stride=2)
        rectified = tools.lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 16, 128, ndf] => [batch, 8, 64, ndf * 2]
    # layer_3: [batch, 8, 64, ndf * 2] => [batch, 4, 32, ndf * 4]
    # layer_4: [batch, 4, 32, ndf * 4] => [batch, 3, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = tools.conv(layers[-1], out_channels, stride=stride)
            normalized = tools.batchnorm(convolved)
            rectified = tools.lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 3, 31, ndf * 8] => [batch, 2, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = tools.conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]
