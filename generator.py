#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:36:31 2017

@author: no1
网络前面写的是现在的，后面的是以前的作为参考
"""
import tensorflow as tf
import tools


def create_generator(generator_inputs, generator_outputs_channels):
    '''
    [<tf.Tensor 'generator/encoder_1/conv/Conv2D:0' shape=(1, 16, 16, 64) dtype=float32>,
    <tf.Tensor 'generator/encoder_2/batchnorm/batchnorm/add_1:0' shape=(1, 8, 8, 128) dtype=float32>,
    <tf.Tensor 'generator/encoder_3/batchnorm/batchnorm/add_1:0' shape=(1, 4, 4, 256) dtype=float32>,
    <tf.Tensor 'generator/encoder_4/batchnorm/batchnorm/add_1:0' shape=(1, 2, 2, 512) dtype=float32>,
    <tf.Tensor 'generator/encoder_5/batchnorm/batchnorm/add_1:0' shape=(1, 1, 1, 512) dtype=float32>,
    <tf.Tensor 'generator/decoder_5/dropout/mul:0' shape=(1, 2, 2, 512) dtype=float32>,
    <tf.Tensor 'generator/decoder_4/batchnorm/batchnorm/add_1:0' shape=(1, 4, 4, 512) dtype=float32>,
    <tf.Tensor 'generator/decoder_3/batchnorm/batchnorm/add_1:0' shape=(1, 8, 8, 256) dtype=float32>,
    <tf.Tensor 'generator/decoder_2/batchnorm/batchnorm/add_1:0' shape=(1, 16, 16, 64) dtype=float32>,
    <tf.Tensor 'generator/decoder_1/Tanh:0' shape=(1, 32, 32, 1) dtype=float32>]
    '''
    layers = []
    ngf=64
    # encoder_1: [batch, 32, 256, in_channels] => [batch, 16, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = tools.conv(generator_inputs, ngf, stride=2)
        layers.append(output)

    layer_specs = [
        ngf * 2, # encoder_2: [batch, 16, 128, ngf] => [batch, 8, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 8, 64, ngf * 2] => [batch, 4, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 4, 32, ngf * 4] => [batch, 2, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 2, 16, ngf * 8] => [batch, 1, 8, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = tools.lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = tools.conv(rectified, out_channels, stride=2)
            output = tools.batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_5: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 ]
        (ngf * 8, 0.0),   # decoder_4: [batch, 2, 2, ngf * 8 ] => [batch, 4, 4, ngf * 8 ]
        (ngf * 4, 0.0),   # decoder_3: [batch, 4, 4, ngf * 8] => [batch, 8, 8, ngf * 4 ]
        (ngf , 0.0),   # decoder_2: [batch, 8, 8, ngf * 4 ] => [batch, 16, 16, ngf ]

    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = tools.deconv(rectified, out_channels)
            output = tools.batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = tools.deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]
