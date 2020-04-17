import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
from scipy import io as sio
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from sklearn import metrics
import os
import shutil
from PIL import Image
import random
import time
import sys
import csv
from modelcls import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

selected_feat = 0, 1, 2, 3, 4,
tasks = 1, 2
model_stats = 'test'
to_restore = False
save_training_images = True
use_syn = True

groups = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 0, 'sSCD': 0, 'pSCD': 1, 'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0,
          'sCN': 0, 'pCN': 0, 'ppCN': 1, 'Autism': 1, 'Control': 0}

output_path = "./output_{0}/ADNIO/".format(tasks)
check_dir = "./output_{0}/adni_ckpts_m-us/".format(tasks)
outputA_path = output_path + "{0}/PET/".format(tasks)
outputB_path = output_path + "{0}/MRI/".format(tasks)
input_path = "D:/ADNI/CycleGAN/" + "input/ADNIO/"
synth_path = "D:/ADNI/CycleGAN/" + "input/cls_dis/"
chkpt_fname = check_dir + '{0}-30'.format(tasks)
max_epoch = 151
max_images = 1000

batch_size = 1
ngf = 16
ndf = 32
argument_side = 3
input_size = [144, 176, 144, 1]
img_width, img_height, img_depth, img_layer = 144, 176, 144, 1
# img_height = 176
# img_depth = 144
# img_layer = 1


class CCGAN():

    def inputAB(self, imdb, cycload=True, augment=True, fullcomb=True):
        flnm, grp = imdb
        if flnm in self.datapool:
            mpdata, label = self.datapool[flnm]
        else:
            label = np.zeros(2, np.float32)
            cls = groups[grp]
            if cls in [0, 1]: label[cls] = 1
            mfile = input_path + 'MRI/' + flnm + '.mat'
            pfile = input_path + 'PET/' + flnm + '.mat'
            smfile = synth_path + 'MRI/' + flnm + '.mat'
            spfile = synth_path + 'PET/' + flnm + '.mat'
            mpdata = []
            for fl in [mfile, pfile, smfile, spfile]:
                if os.path.exists(fl):
                    dt = np.array(sio.loadmat(fl)['IMG'])
                else:
                    dt = None
                    print(fl)
                mpdata.append(dt)
            if cycload:
                self.datapool[flnm] = mpdata, label

        if use_syn:
            if fullcomb:
                # com_list = ((0, 1), (0, 3), (2, 1), (2, 3),)
                com_list = ((0, 3), (2, 1),)
            else:
                com_list = ((0, 3), (0, 1),)
        else:
            com_list = ((0, 1),)

        mdata = []; pdata = []; labels = []
        for idd in com_list:
            if mpdata[idd[0]] is not None and mpdata[idd[1]] is not None:
                if augment:
                    idx = random.randint(-argument_side, argument_side)
                    idy = random.randint(-argument_side, argument_side)
                    idz = random.randint(-argument_side, argument_side)
                else:
                    idx = 0
                    idy = 0
                    idz = 0
                mdata.append(mpdata[idd[0]][18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis])
                pdata.append(mpdata[idd[1]][18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis])
                labels.append(label)
        # print(np.shape(mdata))
        if mdata == []:
            im_m = None
        else:
            im_m = np.minimum(1, np.array(mdata, dtype=np.float32) / 96 - 1.0)
        if pdata == []:
            im_p = None
        else:
            im_p = np.array(pdata, dtype=np.float32) / 128 - 1.0
        # print(pdata == [])
        return im_m, im_p, labels

    def get_database(self, imdbname, vldgrp=("AD", "CN")):
        imdb = []
        with open(imdbname, newline='') as csvfile:
            imdbreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in imdbreader:
                if row[2] in vldgrp:
                    imdb.append(row[1:3])
        return imdb

    def input_setup_adni(self, input_path):
        self.datapool = {}
        self.imdb_train = self.get_database(input_path + '/ADNI1_imdb_36m.csv', ["AD", "CN", 'pMCI', 'sMCI', 'MCI'])
        self.imdb_test  = self.get_database(input_path + '/ADNI2_imdb_36m.csv', ["AD", "CN", 'pMCI', 'sMCI', 'MCI'])
        self.imdb_train = self.imdb_train + self.imdb_test
        self.imdb_test = self.get_database(input_path + '/T2DM_imdb_36m.csv', ['DM', 'CN'])
        print(len(self.imdb_train))
        print(len(self.imdb_test))

    def model_setup(self):
        self.input_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="input_B")
        self.label_holder = tf.placeholder(tf.float32, [None, 2], name="label")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")
        with tf.variable_scope("CLS") as scope:
            self.logit_A, self.prob_A, self.feats_A = build_classifier(self.input_A, 16, order=tasks, name="cls_A")
            self.logit_B, self.prob_B, self.feats_B = build_classifier(self.input_B, 16, order=tasks, name="cls_B")

    def loss_calc(self):
        self.model_vars = tf.trainable_variables()
        for var in self.model_vars: print(var.name)
        self.cls_loss_A = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit_A, labels=self.label_holder))
        self.cls_loss_B = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit_B, labels=self.label_holder))
        optimizer = tf.train.GradientDescentOptimizer(0.01)

        cls_A_vars = [var for var in self.model_vars if 'cls_A' in var.name]
        cls_B_vars = [var for var in self.model_vars if 'cls_B' in var.name]
        self.cls_A_trainer = optimizer.minimize(self.cls_loss_A, var_list=cls_A_vars)
        self.cls_B_trainer = optimizer.minimize(self.cls_loss_B, var_list=cls_B_vars)


    def matrics_calc(self, testvals, labels, pos=0, neg=1):
        mean = np.mean(testvals, axis=0)
        med = np.median(testvals, axis=0)
        thres = np.sort(testvals, axis=0)
        print(thres[len(thres)//9*5])
        # testvals = testvals - thres[len(thres)*5//15]+0.5 # 165/209
        # loss_A: 0.33685695903405627[0.9630564013339132, 0.9064171098758901, 0.896969691533517, 0.9138755937135139, 0.8942593133325786, 0.8103402381484736]
        print(np.sum(testvals, axis=0))
        AUC = metrics.roc_auc_score(y_score=np.transpose(testvals), y_true=np.transpose(labels), average='samples')
        TP = 0; TN=0; FP=0; FN=0
        f = 1.000
        for idx in range(len(testvals)):
            if (labels[idx][pos]*f > labels[idx][neg]) & (testvals[idx][pos]*f > testvals[idx][neg]):
                TP = TP + 1
            if (labels[idx][pos]*f < labels[idx][neg]) & (testvals[idx][pos]*f <= testvals[idx][neg]):
                TN = TN + 1
            if (labels[idx][pos]*f < labels[idx][neg]) & (testvals[idx][pos]*f > testvals[idx][neg]):
                FP = FP + 1
            if (labels[idx][pos]*f > labels[idx][neg]) & (testvals[idx][pos]*f <= testvals[idx][neg]):
                FN = FN + 1

        print(TP, FN, TN, FP)
        ACC = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        SEN = (TP) / (TP + FN + 1e-6)
        SPE = (TN) / (TN + FP + 1e-6)
        PPV = (TP) / (TP + FP + 1e-6)
        F_score = (2 * SEN * PPV) / (SEN + PPV + 1e-6)
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)+ 1e-6)
        return [AUC, ACC, SEN, SPE, F_score, MCC]

    def train(self):
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            if to_restore:
                saver.restore(sess, chkpt_fname)
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            for epoch in range(sess.run(self.global_step), max_epoch):
                print("In the epoch ", epoch)
                saver.save(sess, os.path.join(check_dir, str(tasks)), global_step=epoch)

                curr_lr = 0.0002 * (200-max(100, epoch)) / 100

                trainlabels_A = []; trainprobs_A = []; losses_A = 0
                trainlabels_B = []; trainprobs_B = []; losses_B = 0

                for ptr in range(0, min(max_images, len(self.imdb_train))):
                    # print("In the iteration ", ptr, self.imdb_train[ptr])
                    inputA, inputB, label = self.inputAB(self.imdb_train[ptr], cycload=True, augment=True, fullcomb=True)

                    if inputA is not None:
                        _, prob, loss = sess.run([self.cls_A_trainer, self.prob_A, self.cls_loss_A],
                                                     feed_dict={self.input_A: inputA , self.label_holder: label})
                        losses_A = losses_A + loss; trainlabels_A.append(label); trainprobs_A.append(prob)

                    if inputB is not None:
                        _, prob, loss = sess.run([self.cls_B_trainer, self.prob_B, self.cls_loss_B],
                                                     feed_dict={self.input_B: inputB, self.label_holder: label})
                        losses_B = losses_B + loss; trainlabels_B.append(label); trainprobs_B.append(prob)

                print('loss_A:', losses_A / len(trainprobs_A), self.matrics_calc(np.concatenate(trainprobs_A), np.concatenate(trainlabels_A), pos=1, neg=0))
                print('loss_B:', losses_B / len(trainprobs_B), self.matrics_calc(np.concatenate(trainprobs_B), np.concatenate(trainlabels_B), pos=1, neg=0))
                sess.run(tf.assign(self.global_step, epoch + 1))

    def test(self):
        print("eval the classification results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(30, 45, 1):
                # chkpt_fname = tf.train.latest_checkpoint(check_dir)
                chkpt_fname = check_dir + str(tasks)+'-{0}'.format(epoch)
                print("epoch-{0}".format(epoch), chkpt_fname)
                saver.restore(sess, chkpt_fname)
                testfeats_A = []; testlabels_A = []; testprobs_A = []; losses_A = 0
                testfeats_B = []; testlabels_B = []; testprobs_B = []; losses_B = 0
                reader = tf.train.NewCheckpointReader(chkpt_fname)
                wa = reader.get_tensor('CLS/cls_A/fc_layer/w')
                wb = reader.get_tensor('CLS/cls_B/fc_layer/w')
                rwa = np.reshape(wa[:, 0], [64, -1])
                rwb = np.reshape(wb[:, 0], [64, -1])

                for ptr in range(min(len(self.imdb_test), max_images)):
                    inputA, inputB, label = self.inputAB(self.imdb_test[ptr], cycload=True, augment=False, fullcomb=False)
                    if inputA is not None:
                        prob, loss, feats = sess.run([self.prob_A, self.cls_loss_A, self.feats_A], feed_dict={self.input_A: inputA[0:1], self.label_holder: label[0:1]})
                        losses_A = losses_A + loss; testlabels_A.append(label[0:1]); testprobs_A.append(prob), testfeats_A.append(feats[-1])
                    if inputB is not None:
                        prob, loss, feats = sess.run([self.prob_B, self.cls_loss_B, self.feats_B], feed_dict={self.input_B: inputB[0:1], self.label_holder: label[0:1]})
                        losses_B = losses_B + loss; testlabels_B.append(label[0:1]); testprobs_B.append(prob), testfeats_B.append(feats[-1])
                print('loss_A:', losses_A / len(testprobs_A),
                      self.matrics_calc(np.concatenate(testprobs_A), np.concatenate(testlabels_A), pos=1, neg=0))
                print('loss_B:', losses_B / len(testprobs_B),
                      self.matrics_calc(np.concatenate(testprobs_B), np.concatenate(testlabels_B), pos=1, neg=0))
                print('combination:', self.matrics_calc((np.concatenate(testprobs_A)+np.concatenate(testprobs_B))/2, np.concatenate(testlabels_A), pos=1, neg=0))


def main():
    model = CCGAN()
    model.input_setup_adni(input_path)
    model.model_setup()
    model.loss_calc()
    if model_stats == 'train':
        model.train()
    elif model_stats == 'test':
        model.test()


if __name__ == '__main__':
    main()

