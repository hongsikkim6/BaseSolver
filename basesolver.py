import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import os
import cfg
import time
import pickle
import numpy as np
import glob


class BaseSolver(object):
    def __init__(self, sess, network, tb_dir, log_dir, pretrained_model=None):
        self.net = network
        self.pretrained_model = pretrained_model
        self.log_dir = log_dir
        self.tb_dir = tb_dir
        self.tb_dir_val = tb_dir + '_val'

    def train_model(self, sess, max_iters=cfg.MAX_ITERATION):
        with sess.graph.as_default():
            # EDIT: Delete if don't use fixed random seed
            tf.set_random_seed(cfg.RND_SEED)

            layers = self.net.create_architecture()
            loss = layers['total_loss']

            # EDIT: Define learning rate, optimizer and params
            lr = tf.Variable(cfg.TRAIN_LEARNING_RATE, trainable=False)
            optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN_MOMENTUM)

            gvs = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(gvs)
            self.saver = tf.train.Saver(max_to_keep=100000)

            # EDIT: Delete if don't use tensorboard
            self.writer = tf.summary.FileWriter(self.tb_dir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tb_dir_val)

        lsf, nfiles, sfiles = self.find_snapshots()

        if lsf == 0:
            rate, last_snapshot_iter, np_paths, ss_paths = \
                self.initialize(sess)
        else:
            rate, last_snapshot_iter, np_paths, ss_paths = \
                self.restore(sess, str(sfiles[-1]), str(nfiles[-1]))

        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        while iter < max_iters + 1:
            # EDIT: Delete if no lr reduction
            if iter % cfg.TRAIN_LR_REDUCTION == 0:
                self.snapshot(sess, iter, rate)
                rate *= cfg.TRAIN_LR_GAMMA
                sess.run(tf.assign(lr, rate))

            data = self.next_batch()
            now = time.time()
            if iter == 1 or iter % cfg.TRAIN_SUMMARY_INTERVAL == 0:
                losses, summary = self.net.train_step_with_summary(sess, data, train_op)
                self.writer.add_summary(summary, float(iter))

                # EDIT: Delete if no validations
                val_data = self.next_val_batch()
                summary_val = self.net.get_summary(sess, val_data)
                self.valwriter.add_summary(summary_val, float(iter))

                last_summary_time = now
            else:
                losses = self.net.train_step(sess, data, train_op)

            if iter % (cfg.TRAIN_DISPLAY) == 0:
                # EDIT: Display EDIT
                print(iter, losses)

            if iter == 1 or iter % (cfg.TRAIN_SNAPSHOT_ITER) == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess, iter, rate)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                if len(np_paths) > cfg.TRAIN_SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)
            iter += 1

        self.writer.close()
        self.valwriter.close()

    def initialize(self, sess):
        np_paths = []
        ss_paths = []
        last_snapshot_iter = 0
        rate = cfg.TRAIN_LEARNING_RATE

        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))

        # EDIT: want to use pretrained model, use below
        print('Loading initial model weights from pretrained model')
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)
        print('Loaded')

        return rate, last_snapshot_iter, np_paths, ss_paths

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(str(e))

    def restore(self, sess, sfile, nfile):
        np_paths = [nfile]
        ss_paths = [sfile]
        last_snapshot_iter, last_snapshot_rate = self.from_snapshot(sess, sfile, nfile)

        return last_snapshot_rate, last_snapshot_iter, np_paths, ss_paths

    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored')

        # Edit: use nfile to save and restore hyperparameters
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            vcur = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)
            last_snapshot_rate = pickle.load(fid)

        np.random.set_state(st0)
        self.cur = cur
        self.vcur = vcur

        return last_snapshot_iter, last_snapshot_rate

    def snapshot(self, sess, iter, rate):
        net = self.net
        filename = cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + 'ckpt'
        filename = os.path.join(self.log_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # EDIT: use nfile to save and restore hyperparameters
        nfilename = cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + 'pkl'
        nfilename = os.path.join(self.log_dir, nfilename)

        st0 = np.random.get_state()
        cur = self.cur
        vcur = self.vcur
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vcur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(rate, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def find_snapshots(self):
        # EDIT: this will automatically find the most recent snapshots.
        sfiles = os.path.join(self.log_dir, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)

        nfiles = [ss.replace('.ckpt.meta', 'pkl') for ss in sfiles]
        sfiles = [ss.replace('.meta', '') for ss in sfiles]

        lsf = len(sfiles)
        assert lsf == len(nfiles)

        return lsf, nfiles, sfiles

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(ss_paths) - cfg.TRAIN_SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            os.remove(str(sfile + '.meta'))
            ss_paths.remove(sfile)

        # EDIT: remove hyperparameter nfiles
        to_remove = len(np_paths) - cfg.TRAIN_SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

    def next_batch(self):
        raise NotImplementedError
