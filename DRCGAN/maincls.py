import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
from scipy import io as sio
from scipy import ndimage
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
tasks = 1, 2,
model_stats = 'test'
to_restore = False
save_training_images = True
use_syn = False

groups = {'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 1, 'sSCD': 0, 'pSCD': 1, 'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0,
          'sCN': 0, 'pCN': 0, 'ppCN': 1, 'Autism': 1, 'Control': 0}
grpset1 = ["CN", 'pMCI', 'sMCI', 'MCI']
grpset2 = ["AD", "CN"]
grpset3 = ["sSCD", "pSCD"]

output_path = "./output_{0}/ADNIO/".format(tasks)
if use_syn:
    check_dir = "./output_{0}/adni_ckpts-us/".format(tasks)
else:
    check_dir = "./output_{0}/adni_ckpts/".format(tasks)
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
img_width = 144
img_height = 176
img_depth = 144
img_layer = 1


class CCGAN():

    def inputAB(self, imdb, cycload=True, augment=True):
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
                    # print(fl)
                mpdata.append(dt)
            if cycload:
                self.datapool[flnm] = mpdata, label

        if augment:
            idx = random.randint(-argument_side, argument_side)
            idy = random.randint(-argument_side, argument_side)
            idz = random.randint(-argument_side, argument_side)
        else:
            idx = 0
            idy = 0
            idz = 0
        mdata = []; pdata = []; labels = []
        if use_syn:
            if mpdata[0] is not None:
                mdata.append(mpdata[0][18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis])
                pdata.append(mpdata[3][18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis])
                labels.append(label)
            if mpdata[1] is not None:
                mdata.append(mpdata[2][18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis])
                pdata.append(mpdata[1][18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis])
                labels.append(label)
        else:
            if mpdata[0] is not None:
                mdata.append(mpdata[0][18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis])
            if mpdata[1] is not None:
                pdata.append(mpdata[1][18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis])
            labels.append(label)
        # if use_syn:
        #     if mpdata[0] is not None:
        #         mdata.append(mpdata[0][2 + idx:178 + idx, 4 + idy:212 + idy, 2 + idz:178 + idz, np.newaxis])
        #         pdata.append(mpdata[3][2 + idx:178 + idx, 4 + idy:212 + idy, 2 + idz:178 + idz, np.newaxis])
        #         labels.append(label)
        #     if mpdata[1] is not None:
        #         mdata.append(mpdata[2][2 + idx:178 + idx, 4 + idy:212 + idy, 2 + idz:178 + idz, np.newaxis])
        #         pdata.append(mpdata[1][2 + idx:178 + idx, 4 + idy:212 + idy, 2 + idz:178 + idz, np.newaxis])
        #         labels.append(label)
        # else:
        #     if mpdata[0] is not None:
        #         mdata.append(mpdata[0][2 + idx:178 + idx, 4 + idy:212 + idy, 2 + idz:178 + idz, np.newaxis])
        #     if mpdata[1] is not None:
        #         pdata.append(mpdata[1][2 + idx:178 + idx, 4 + idy:212 + idy, 2 + idz:178 + idz, np.newaxis])
        #     labels.append(label)
        if mdata == []:
            im_m = None
        else:
            im_m = np.minimum(1, np.array(mdata, dtype=np.float32) / 96 - 1.0)
        if pdata == []:
            im_p = None
        else:
            im_p = np.array(pdata, dtype=np.float32) / 128 - 1.0
        return im_m, im_p, labels

    def get_database(self, imdbname, vldgrp=("AD", "CN")):
        imdb = []
        with open(imdbname, newline='') as csvfile:
            imdbreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in imdbreader:
                if row[2] in vldgrp:
                    if row[1:3] in imdb:
                        print(row[1:3])
                    imdb.append(row[1:3])
        return imdb

    def input_setup_adni(self, input_path):
        self.datapool = {}
        self.imdb_train = self.get_database(input_path + '/ADNI1_imdb_36m.csv', grpset1)
        self.imdb_test  = self.get_database(input_path + '/ADNI2_imdb_36m.csv', grpset2)
        self.imdb_test = self.get_database(input_path + '/AIBL_imdb_36m.csv', grpset2)
        self.imdb_test = self.get_database(input_path + '/sSCDpSCD.csv', grpset3)
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
        f = 1.70
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
                if (epoch < 100):
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002 * (epoch - 100) / 50

                trainlabels_A = []; trainprobs_A = []; losses_A = 0
                trainlabels_B = []; trainprobs_B = []; losses_B = 0

                for ptr in range(0, min(max_images, len(self.imdb_train))):
                    # print("In the iteration ", ptr, self.imdb_train[ptr])
                    inputA, inputB, label = self.inputAB(self.imdb_train[ptr], cycload=True, augment=True)

                    if inputA is not None:
                        _, prob, loss = sess.run([self.cls_A_trainer, self.prob_A, self.cls_loss_A],
                                                     feed_dict={self.input_A: inputA, self.label_holder: label})
                        losses_A = losses_A + loss; trainlabels_A.append(label); trainprobs_A.append(prob)

                    if inputB is not None:
                        _, prob, loss = sess.run([self.cls_B_trainer, self.prob_B, self.cls_loss_B],
                                                     feed_dict={self.input_B: inputB, self.label_holder: label})
                        losses_B = losses_B + loss; trainlabels_B.append(label); trainprobs_B.append(prob)

                print('loss_A:', losses_A / (len(trainprobs_A)+1e-6), self.matrics_calc(np.concatenate(trainprobs_A), np.concatenate(trainlabels_A), pos=1, neg=0))
                print('loss_B:', losses_B / (len(trainprobs_B)+1e-6), self.matrics_calc(np.concatenate(trainprobs_B), np.concatenate(trainlabels_B), pos=1, neg=0))
                sess.run(tf.assign(self.global_step, epoch + 1))

    def test(self):
        print("eval the classification results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(7, 151, 1):
                # chkpt_fname = tf.train.latest_checkpoint(check_dir)

                chkpt_fname = check_dir + str(tasks)+'-{0}'.format(epoch)
                print("epoch-{0}".format(epoch), chkpt_fname)
                saver.restore(sess, chkpt_fname)
                testfeats_A = []; testlabels_A = []; testprobs_A = []; losses_A = 0
                testfeats_B = []; testlabels_B = []; testprobs_B = []; losses_B = 0

                for ptr in range(min(len(self.imdb_test), max_images)):
                    inputA, inputB, label = self.inputAB(self.imdb_test[ptr], cycload=True, augment=False)
                    if inputA is not None:
                        # print(self.imdb_test[ptr])
                        prob, loss, feats = sess.run([self.prob_A, self.cls_loss_A, self.feats_A], feed_dict={self.input_A: inputA[0:1], self.label_holder: label[0:1]})
                        losses_A = losses_A + loss; testlabels_A.append(label[0:1]); testprobs_A.append(prob), testfeats_A.append(feats[-1])
                    if inputB is not None:
                        prob, loss, feats = sess.run([self.prob_B, self.cls_loss_B, self.feats_B], feed_dict={self.input_B: inputB[0:1], self.label_holder: label[0:1]})
                        losses_B = losses_B + loss; testlabels_B.append(label[0:1]); testprobs_B.append(prob), testfeats_B.append(feats[-1])
                print('loss_A:', losses_A / (len(testprobs_A)+1e-9),
                      self.matrics_calc(np.concatenate(testprobs_A), np.concatenate(testlabels_A), pos=1, neg=0))
                # print('loss_B:', losses_B / (len(testprobs_B)+1e-9),
                #       self.matrics_calc(np.concatenate(testprobs_B), np.concatenate(testlabels_B), pos=1, neg=0))
                # print('combination:', self.matrics_calc((np.concatenate(testprobs_A)+np.concatenate(testprobs_B))/2, np.concatenate(testlabels_A), pos=1, neg=0))

    def showregions(self):
        print("eval weight classification results")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        inputA, inputB, label = self.inputAB(self.imdb_test[100], cycload=True, augment=False)
        mfile = input_path + 'MRI/' + self.imdb_test[100][0] + '.mat'
        pfile = input_path + 'PET/' + self.imdb_test[100][0] + '.mat'
        inputA = np.array(sio.loadmat(mfile)['IMG'], np.float32)/96.0-1.0
        inputB = np.array(sio.loadmat(pfile)['IMG'], np.float32)/128.0-1.0
        print(np.shape(inputA), np.shape(inputB))
        for epoch in range(80, 131, 1):
            chkpt_fname = check_dir + str(tasks)+'-{0}'.format(epoch)
            print("epoch-{0}".format(epoch), chkpt_fname)

            reader = tf.train.NewCheckpointReader(chkpt_fname)
            wa = reader.get_tensor('CLS/cls_A/fc_layer/w')
            wb = reader.get_tensor('CLS/cls_B/fc_layer/w')
            rwa = np.reshape(wa[:, 0], [-1, 128])
            rwb = np.reshape(wb[:, 0], [-1, 128])
            wra = np.reshape(np.sum(rwa * rwa, axis=1), [4, 5, 4])
            wrb = np.reshape(np.sum(rwb * rwb, axis=1), [4, 5, 4])
            mask = np.zeros((181, 217, 181), np.float32)+0.000001
            mask[42:162:32, 46:198:32, 34:154:32] = wra
            maskA = ndimage.gaussian_filter(mask, 16.0) * 256
            mask[42:162:32, 46:198:32, 34:154:32] = wrb
            maskB = ndimage.gaussian_filter(mask, 16.0) * 256
            sio.savemat('cls{0}_AB.mat'.format(epoch), {'mA': maskA, 'mB': maskB, 'iA': inputA, 'iB': inputB})
            # np.savetxt('cls{0}_A2.txt'.format(epoch), wra, '%.10f')
            # np.savetxt('cls{0}_B2.txt'.format(epoch), wra, '%.10f')


def main():
    model = CCGAN()
    model.input_setup_adni(input_path)
    model.model_setup()
    model.loss_calc()
    if model_stats == 'train':
        model.train()
    elif model_stats == 'test':
        model.test()
    elif model_stats == 'showregions':
        model.showregions()


if __name__ == '__main__':
    main()