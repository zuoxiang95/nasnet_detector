# -*- coding: utf-8 -*-

from nasnet.nasnet import NASNetLarge, classifier, rpn
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from nasnet import losses as losses
import nasnet.roi_helpers as roi_helpers
from keras.utils import generic_utils


def train():
    img_input = Input(shape=(331, 331))
    roi_input = Input(shape=(None, 4))
    base_model = NASNetLarge(input_shape=(331, 331, 3), weights='imagenet', include_top=False, pooling='avg')
    conv_feature = base_model(img_input)

    num_anchors = 10
    nas_rpn = rpn(conv_feature, num_anchors)

    nas_classifier = classifier(conv_feature, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], nas_classifier)

    model_all = Model([img_input, roi_input], rpn[:2] + nas_classifier)

    # load weight
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = 1000
    num_epochs = int(options.num_epochs)
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf
    class_mapping_inv = {v: k for k, v in class_mapping.items()}

    print('Starting training')

    vis = True
