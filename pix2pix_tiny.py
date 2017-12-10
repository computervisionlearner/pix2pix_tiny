#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:26:29 2017

@author: no1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
import json
import math
import time
import tools
from _model import create_model
from reader import load_examples
import logging
from datetime import datetime
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",type=str,default='facades/train', help="path to folder containing images")
parser.add_argument("--mode", type=str, default='train', choices=["train", "test", "export"])
parser.add_argument("--output_dir", type=str ,default='facades_train', help="where to put output files")

parser.add_argument("--checkpoint", default='checkpoint', help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=200,help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="BtoA", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=32, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
'''
这一版本把export功能给删掉了
'''
# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
FLAGS = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 32
def initLogging(logFilename='record.log'):
    """Init for logging
    """
    logging.basicConfig(
                      level    = logging.DEBUG,
                      format='%(asctime)s-%(levelname)s-%(message)s',
                      datefmt  = '%y-%m-%d %H:%M',
                      filename = logFilename,
                      filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

initLogging()

def save_images(fetches, step=None):
    image_dir = os.path.join(FLAGS.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(FLAGS.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")


    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if FLAGS.mode == "test" or FLAGS.mode == "export":
        if FLAGS.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        FLAGS.flip = False

    for k, v in FLAGS._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(FLAGS.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(FLAGS), sort_keys=True, indent=4))

    with tf.Graph().as_default():
        examples = load_examples(FLAGS)
        print("examples count = %d" % examples.count)

        # inputs and targets are [batch_size, height, width, channels]
        model = create_model(examples.inputs, examples.targets, FLAGS)

        inputs = tools.deprocess(examples.inputs) #[-1,1]--->(0,1)
        targets = tools.deprocess(examples.targets)
        outputs = tools.deprocess(model.outputs)
        train_op = model.train
        def convert(image):
            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

        # reverse any processing on images so they can be written to disk or displayed to user
        with tf.name_scope("convert_inputs"):
            converted_inputs = convert(inputs)

        with tf.name_scope("convert_targets"):
            converted_targets = convert(targets)

        with tf.name_scope("convert_outputs"):
            converted_outputs = convert(outputs)

        with tf.name_scope("encode_images"):
            display_fetches = {
                "paths": examples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            }

        # summaries
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", converted_inputs)

        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", converted_targets)

        with tf.name_scope("outputs_summary"):
            tf.summary.image("outputs", converted_outputs)

        with tf.name_scope("predict_real_summary"):
            tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

        with tf.name_scope("predict_fake_summary"):
            tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

        tf.summary.scalar("discriminator_loss", model.discrim_loss)
        tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
        tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        summary = tf.summary.merge_all()
        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()

        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)
        max_step = examples.count
        if FLAGS.mode == 'train':
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                print('all parameters = ',sess.run(parameter_count))
                for epoch in range(FLAGS.max_epochs):
                    for step in range(max_step):
                        totol_step = step + epoch * max_step + 1
                        start_time = time.time()
                        sess.run(train_op)
                        duration = time.time() - start_time
                        if totol_step % 10 == 0:

                            fetches = {
                                    "train": model.train,
                                    "discrim_loss" : model.discrim_loss,
                                    "gen_loss_GAN" : model.gen_loss_GAN,
                                    "gen_loss_L1" : model.gen_loss_L1,
                                    "display_byte": display_fetches
                                    }
                            results = sess.run(fetches)
                            summary_str = sess.run(summary)
                            summary_writer.add_summary(summary_str, totol_step)
                            summary_writer.flush()
                            logging.info('>>progress epoch %d, global step %d, discrim_loss = %.2f, gen_loss_GAN = %.2f, gen_loss_L1 = %.2f (%.3f sec)'
                            % (epoch, totol_step, results["discrim_loss"], results["gen_loss_GAN"], results["gen_loss_L1"],duration))

                          #-------------------------------

                        if totol_step % 100 == 0 :
                            filesets = save_images(results["display_byte"], step = totol_step)
                            append_index(filesets, step=True)
                        if totol_step % 1000 == 0:
                            logging.info('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
                            saver.save(sess, os.path.join(FLAGS.checkpoint,'ckpt'), global_step=totol_step)


            except KeyboardInterrupt:
                print('INTERRUPTED')

            finally:
                saver.save(sess, os.path.join(FLAGS.checkpoint,'ckpt'), global_step = totol_step)
                print('Model saved in file :%s'% FLAGS.checkpoint)
                coord.request_stop()
                coord.join(threads)
        elif FLAGS.mode == 'test':
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for step in range(max_step):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            coord.request_stop()
            coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()

