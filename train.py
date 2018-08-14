#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation, get_ids
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################

    # parse annotations of the training set
    sets_from_2007 = [('2007', 'train'), ('2007', 'val')]
    train_set = [('2012', 'train')]
    val_set = [('2012', 'val')]
    test_set = [('2007', 'test')]
    
    voc_path ='/media/eHD/datasets/pascalVOC2012/VOCtrain/VOC2012/'
    test_path = '/media/eHD/datasets/pascalVOC2007/VOC_test/'
    train_ids = get_ids(voc_path, train_set)
    val_ids = get_ids(voc_path, val_set)
    test_ids = get_ids(test_path, test_set)
    train_ids_2007 = get_ids('/media/eHD/datasets/pascalVOC2007/VOC_train/', sets_from_2007)

    train_imgs_2007, seen_train_labels_2007 = parse_annotation('/media/eHD/datasets/pascalVOC2007/VOC_train/', 
                                                           '/media/eHD/datasets/pascalVOC2007/VOC_train/JPEGImages/', 
                                                           train_ids_2007, config['model']['labels'])
    train_imgs_2012, seen_train_labels_2012 = parse_annotation(voc_path, '/media/eHD/datasets/pascalVOC2012/VOCtrain/VOC2012/JPEGImages/', 
                                                           train_ids, config['model']['labels'])
    train_imgs=train_imgs_2007+train_imgs_2012
    train_labels=seen_train_labels_2012
    
    valid_imgs, seen_valid_labels = parse_annotation(voc_path, '/media/eHD/datasets/pascalVOC2012/VOCtrain/VOC2012/JPEGImages/', 
                                                    val_ids, config['model']['labels'])
      
    test_imgs, seen_test_labels = parse_annotation(test_path, 
                                               '/media/eHD/datasets/pascalVOC2007/VOC_test/JPEGImages/', 
                                                test_ids, config['model']['labels'])
    print('Number of train images:', len(train_imgs))
    print('Number of validation images:', len(valid_imgs))
    print('Number of test images:', len(test_imgs))
    
    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        
    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the warmup process 
    ###############################

    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               test_imgs          = test_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epochs          = 0, 
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = 3,
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = config['train']['debug'])
    
    ###############################
    #   Start the training process 
    ###############################

    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               test_imgs          = test_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epochs          = 100, 
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = 0,
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = config['train']['debug'])


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
