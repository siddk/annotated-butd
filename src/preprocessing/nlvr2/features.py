"""
features.py

Reads in a tsv file with pre-trained bottom up attention features and writes them to hdf5 file. Additionally builds
image ID --> Feature IDX Mapping.

Hierarchy of HDF5 file:
    { 'image_features': num_images x num_boxes x 2048 array of features
      'image_bb': num_images x num_boxes x 4 array of bounding boxes }

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/tools/detection_features_converter.py
"""
import base64
import csv
import h5py
import numpy as np
import os
import pickle
import sys

# Set CSV Field Size Limit (Big TSV Files...)
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf", "attrs_id", "attrs_conf", "num_boxes", "boxes",
              "features"]
NUM_FIXED_BOXES = 36
FEATURE_LENGTH = 2048


def nlvr2_create_image_features(nlvr2_f='data/NLVR2-Features', cache='data/NLVR2-Cache'):
    """ Iterate through BUTD TSV and Build HDF5 Files with Bounding Box Features, Image ID --> IDX Mappings """
    print('\t[*] Setting up HDF5 Files for Image/Object Features...')

    # Create Trackers for Image IDX --> Index
    train_indices, dev_indices, test_indices = {}, {}, {}
    tfile = os.path.join(cache, 'train36.hdf5')
    dfile = os.path.join(cache, 'dev36.hdf5')
    tsfile = os.path.join(cache, 'test36.hdf5')

    tidxfile = os.path.join(cache, 'train36_img2idx.pkl')
    didxfile = os.path.join(cache, 'dev36_img2idx.pkl')
    tsidxfile = os.path.join(cache, 'test36_img2idx.pkl')

    # Shortcut --> Based on if Files Exist
    if os.path.exists(tfile) and os.path.exists(dfile) and os.path.exists(tsfile) and \
            os.path.exists(tidxfile) and os.path.exists(didxfile) and os.path.exists(tsidxfile):

        with open(tidxfile, 'rb') as f:
            train_indices = pickle.load(f)

        with open(didxfile, 'rb') as f:
            dev_indices = pickle.load(f)

        with open(tsidxfile, 'rb') as f:
            test_indices = pickle.load(f)

        return train_indices, dev_indices, test_indices

    with h5py.File(tfile, 'w') as h_train, h5py.File(dfile, 'w') as h_dev, h5py.File(tsfile, 'w') as h_test:
        # Get Number of Images in each Split
        with open(os.path.join(nlvr2_f, 'train_obj36.tsv'), 'r') as f:
            ntrain = len(f.readlines())

        with open(os.path.join(nlvr2_f, 'valid_obj36.tsv'), 'r') as f:
            ndev = len(f.readlines())

        with open(os.path.join(nlvr2_f, 'test_obj36.tsv'), 'r') as f:
            ntest = len(f.readlines())

        # Setup HDF5 Files
        train_img_features = h_train.create_dataset('image_features', (ntrain, NUM_FIXED_BOXES, FEATURE_LENGTH), 'f')
        train_img_bb = h_train.create_dataset('image_bb', (ntrain, NUM_FIXED_BOXES, 4), 'f')
        train_spatial_features = h_train.create_dataset('spatial_features', (ntrain, NUM_FIXED_BOXES, 6), 'f')

        dev_img_features = h_dev.create_dataset('image_features', (ndev, NUM_FIXED_BOXES, FEATURE_LENGTH), 'f')
        dev_img_bb = h_dev.create_dataset('image_bb', (ndev, NUM_FIXED_BOXES, 4), 'f')
        dev_spatial_features = h_dev.create_dataset('spatial_features', (ndev, NUM_FIXED_BOXES, 6), 'f')

        test_img_features = h_test.create_dataset('image_features', (ntest, NUM_FIXED_BOXES, FEATURE_LENGTH), 'f')
        test_img_bb = h_test.create_dataset('image_bb', (ntest, NUM_FIXED_BOXES, 4), 'f')
        test_spatial_features = h_test.create_dataset('spatial_features', (ntest, NUM_FIXED_BOXES, 6), 'f')

        # Start Iterating through TSV
        print('\t[*] Reading Train TSV File and Populating HDF5 File...')
        train_counter, dev_counter, test_counter = 0, 0, 0
        with open(os.path.join(nlvr2_f, 'train_obj36.tsv'), 'r') as tsv:
            reader = csv.DictReader(tsv, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['num_boxes'] = int(item['num_boxes'])
                image_id = item['img_id']
                image_w = float(item['img_w'])
                image_h = float(item['img_h'])
                bb = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape((item['num_boxes'], -1))

                box_width = bb[:, 2] - bb[:, 0]
                box_height = bb[:, 3] - bb[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bb[:, 0] / image_w
                scaled_y = bb[:, 1] / image_h

                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                     scaled_y,
                     scaled_x + scaled_width,
                     scaled_y + scaled_height,
                     scaled_width,
                     scaled_height),
                    axis=1)

                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bb
                train_img_features[train_counter, :, :] = \
                    np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_features[train_counter, :, :] = spatial_features
                train_counter += 1

        print('\t[*] Reading Dev TSV File and Populating HDF5 File...')
        with open(os.path.join(nlvr2_f, 'valid_obj36.tsv'), 'r') as tsv:
            reader = csv.DictReader(tsv, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['num_boxes'] = int(item['num_boxes'])
                image_id = item['img_id']
                image_w = float(item['img_w'])
                image_h = float(item['img_h'])
                bb = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(
                    (item['num_boxes'], -1))

                box_width = bb[:, 2] - bb[:, 0]
                box_height = bb[:, 3] - bb[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bb[:, 0] / image_w
                scaled_y = bb[:, 1] / image_h

                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                     scaled_y,
                     scaled_x + scaled_width,
                     scaled_y + scaled_height,
                     scaled_width,
                     scaled_height),
                    axis=1)

                dev_indices[image_id] = dev_counter
                dev_img_bb[dev_counter, :, :] = bb
                dev_img_features[dev_counter, :, :] = \
                    np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(
                        (item['num_boxes'], -1))
                dev_spatial_features[dev_counter, :, :] = spatial_features
                dev_counter += 1

        print('\t[*] Reading Test TSV File and Populating HDF5 File...')
        with open(os.path.join(nlvr2_f, 'test_obj36.tsv'), 'r') as tsv:
            reader = csv.DictReader(tsv, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['num_boxes'] = int(item['num_boxes'])
                image_id = item['img_id']
                image_w = float(item['img_w'])
                image_h = float(item['img_h'])
                bb = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(
                    (item['num_boxes'], -1))

                box_width = bb[:, 2] - bb[:, 0]
                box_height = bb[:, 3] - bb[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bb[:, 0] / image_w
                scaled_y = bb[:, 1] / image_h

                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                     scaled_y,
                     scaled_x + scaled_width,
                     scaled_y + scaled_height,
                     scaled_width,
                     scaled_height),
                    axis=1)

                test_indices[image_id] = test_counter
                test_img_bb[test_counter, :, :] = bb
                test_img_features[test_counter, :, :] = \
                    np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(
                        (item['num_boxes'], -1))
                test_spatial_features[test_counter, :, :] = spatial_features
                test_counter += 1

    # Dump Train and Validation Indices to File
    with open(tidxfile, 'wb') as f:
        pickle.dump(train_indices, f)

    with open(didxfile, 'wb') as f:
        pickle.dump(dev_indices, f)

    with open(tsidxfile, 'wb') as f:
        pickle.dump(test_indices, f)

    return train_indices, dev_indices, test_indices
