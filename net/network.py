import tensorflow as tf


class Network(object):
    def __init__(self):
        self._losses = {}


    def create_architecture(self):
        raise NotImplementedError

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def train_step(self, sess, data, train_op):
        raise NotImplementedError

    def train_step_with_summary(self, sess, data, train_op):
        raise NotImplementedError

    def get_summary(self, val_data):
        raise NotImplementedError

