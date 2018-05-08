import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import subprocess
import random


class Imagenet2012(object):

    def __init__(self, path, batch_size, shuffle=None):
        self._gt_class_path = '/home/hongsikkim/HDD/data/Imagenet/bbox/train'
        self._classes = self._get_gt_classes()
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(
            list(
                zip(self._classes, list(range(self._num_classes)))
            )
        )
        self._path = os.path.join(path, 'bbox')
        self._img_path = path
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._image_ext = '.JPEG'
        self._train_path = os.path.join(self._path, 'train')
        self._val_path = os.path.join(self._path, 'val')
        self._train_list = self._get_train_list()
        self._val_list = self._get_val_list()
        self._train_queue = list()
        self._val_queue = list()

    def _get_gt_classes(self):
        filename = os.path.join(self._gt_class_path, 'Imagenet_gt_classes.pkl')
        with open(filename, 'rb') as f:
            _classes = pickle.load(f)

        return _classes

    def _get_train_list(self):
        _savefile = os.path.join(self._img_path, 'train_list.pkl')
        if os.path.exists(_savefile):
            print('Predefined train_list found. Loading...')
            with open(_savefile, 'rb') as fid:
                _train_list = pickle.load(fid)
            print('Done.')
            return _train_list

        _train_list = dict()
        _train_dir = os.listdir(self._train_path)
        for _subdir in _train_dir:
            if '.' in _subdir:
                continue
            _files = os.listdir(os.path.join(self._train_path, _subdir))
            for _file in _files:
                _filename = os.path.join(self._train_path, _subdir, _file)
                _img_name = os.path.join(
                    self._img_path, 'train', _subdir,
                    _file.split('.')[0] + '.JPEG')
                _train_list[_img_name] = self._load_annotation(_filename)
        print('Saving train_list for future use')
        with open(_savefile, 'wb') as fid:
            pickle.dump(_train_list, fid, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done')
        return _train_list

    def _get_val_list(self):
        _savefile = os.path.join(self._img_path, 'val_list.pkl')
        if os.path.exists(_savefile):
            print('Predefined val_list found. Loading...')
            with open(_savefile, 'rb') as fid:
                _val_list = pickle.load(fid)
            print('Done.')
            return _val_list

        _val_list = dict()
        _files = os.listdir(self._val_path)
        for _file in _files:
            _filename = os.path.join(self._val_path, _file)
            _img_name = os.path.join(
                self._img_path, 'val',
                _file.split('.')[0] + '.JPEG')
            _val_list[_img_name] = self._load_annotation(_filename)
        print('Saving val_list for future use')
        with open(_savefile, 'wb') as fid:
            pickle.dump(_val_list, fid, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done')
        return _val_list

    def get_next_train_batch(self):
        _batch = list()
        if len(self._train_queue) < self._batch_size:
            self._train_queue.extend(self._train_list.keys())
        random.shuffle(self._train_queue)
        for _ in range(self._batch_size):
            _batch.append(self._train_queue.pop())

        print(_batch[0])
        print(self._train_list[_batch[0]])
        return _batch

    def get_next_val_batch(self):
        _batch = list()
        if len(self._val_queue) < self._batch_size:
            self._val_queue.extend(self._val_list.keys())
        random.shuffle(self._val_queue)
        for _ in range(self._batch_size):
            _batch.append(self._val_queue.pop())

        print(_batch[0])
        print(self._val_list[_batch[0]])
        return _batch

    def _load_annotation(self, index):
        filename = os.path.join(index)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self._num_classes), dtype=np.float32)
        seg_areas = np.zeros(num_objs, dtype=np.float32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            try:
                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            except:
                try:
                    cls = \
                        self._class_to_ind[
                            index.split('/')[-1].split('_')[0].lower().strip()]
                except:
                    cls = self._class_to_ind['__background__']
                    print(index)

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}


if __name__ == '__main__':
    test = Imagenet2012(
        path='/home/hongsikkim/HDD/data/Imagenet',
        batch_size=4,
        shuffle=None)
    for i in range(5):
        test.get_next_train_batch()
        #test.get_next_val_batch()