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
from layers import *
from modelccgan import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

selected_feat = 0, 1, 2, 3, 4,
tasks = 'cls', 'dis',
model_stats = 'test'
to_restore = False
save_training_images = True
use_mask = False

groups = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 0, 'sSCD': 0, 'pSCD': 1, 'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0,
          'sCN': 0, 'pCN': 0, 'ppCN': 1, 'Autism': 1, 'Control': 0}

output_path = "./output_{0}_fm{1}/ADNIO/".format(tasks, selected_feat)
check_dir = "./output_{0}_fm{1}/adni_ckpts/".format(tasks, selected_feat)
outputA_path = output_path + "{0}/PET/".format(tasks)
outputB_path = output_path + "{0}/MRI/".format(tasks)
input_path = "D:/ADNI/CycleGAN/" + "input/ADNIO/"

chkpt_fname = check_dir + '{0}-100'.format(tasks)
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
            mdata, pdata, label = self.datapool[flnm]
        else:
            label = np.zeros(2, np.float32)
            cls = groups[grp]
            if cls in [0, 1]: label[cls] = 1
            mfile = 'MRI/' + flnm + '.mat'
            pfile = 'PET/' + flnm + '.mat'

            if os.path.exists(input_path + mfile):
                mdata = np.array(sio.loadmat(input_path + mfile)['IMG'])
            else:
                mdata = None
                print(mfile)
            if os.path.exists(input_path + pfile):
                pdata = np.array(sio.loadmat(input_path + pfile)['IMG'])
            else:
                pdata = None
                print(pfile)
            if cycload:
                self.datapool[flnm] = mdata, pdata, label

        if augment:
            idx = random.randint(-argument_side, argument_side)
            idy = random.randint(-argument_side, argument_side)
            idz = random.randint(-argument_side, argument_side)
        else:
            idx = 0
            idy = 0
            idz = 0

        if mdata is None:
            im_m = None
        else:
            im_m = mdata[np.newaxis, 18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis]
            im_m = np.minimum(1, im_m.astype(np.float32) / 96 - 1.0)
        if pdata is None:
            im_p = None
        else:
            im_p = pdata[np.newaxis, 18 + idx:162 + idx, 22 + idy:198 + idy, 10 + idz:154 + idz, np.newaxis]
            im_p = im_p.astype(np.float32) / 128 - 1.0
        labels = label[np.newaxis, :]
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
        # self.imdb_test  = self.get_database(input_path + '/ADNI2_imdb_36m.csv', ["AD", "CN", 'pMCI', 'sMCI', 'MCI'])
        # self.imdb_test = self.get_database(input_path + '/AIBL_imdb_36m.csv', ['AD', 'CN', 'pMCI', 'sMCI', 'MCI'])
        self.imdb_test = self.get_database(input_path + '/T2DM_imdb.csv', ['DM', 'CN', 'pMCI', 'sMCI', 'MCI'])
        print(len(self.imdb_train))
        print(len(self.imdb_test))

    def model_setup(self):
        self.input_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="input_B")
        self.label_holder = tf.placeholder(tf.float32, [None, 2], name="label")

        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="fake_pool_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("GAN") as scope:
            self.fake_B = build_generator(self.input_A, ngf, numofres=3, name="g_A")
            self.fake_A = build_generator(self.input_B, ngf, numofres=3, name="g_B")
            self.rec_A = build_gen_discriminator(self.input_A, ndf, "d_A")
            self.rec_B = build_gen_discriminator(self.input_B, ndf, "d_B")

            scope.reuse_variables()
            self.fake_rec_A = build_gen_discriminator(self.fake_A, ndf, "d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B, ndf, "d_B")
            self.cyc_A = build_generator(self.fake_B, ngf, numofres=3, name="g_B")
            self.cyc_B = build_generator(self.fake_A, ngf, numofres=3, name="g_A")
            scope.reuse_variables()
            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, ndf, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, ndf, "d_B")

        with tf.variable_scope("CLS") as scope:
            self.logit_A, self.prob_A, self.feats_A = build_classifier(self.input_A, 16, backbone='vgg3d', name="cls_A")
            self.logit_B, self.prob_B, self.feats_B = build_classifier(self.input_B, 16, backbone='vgg3d', name="cls_B")
            scope.reuse_variables()
            self.logit_fake_A, self.prob_fake_A, self.feats_fake_A = build_classifier(self.fake_A, 16, backbone='vgg3d', name="cls_A")
            self.logit_fake_B, self.prob_fake_B, self.feats_fake_B = build_classifier(self.fake_B, 16, backbone='vgg3d', name="cls_B")


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
        if use_mask:
            msk_A = tf.cast(self.input_A > - 1, tf.float32)
            msk_B = tf.cast(self.input_B > - 1, tf.float32)
        else:
            msk_A = 1
            msk_B = 1
        cls_loss_A = 0; cls_loss_B = 0
        for slt in selected_feat:
            cls_loss_A = cls_loss_A + tf.reduce_mean(tf.abs(self.feats_fake_A[slt] - self.feats_A[slt]))
            cls_loss_B = cls_loss_B + tf.reduce_mean(tf.abs(self.feats_fake_B[slt] - self.feats_B[slt]))
        cyc_loss_A = tf.reduce_mean(tf.abs(self.input_A - self.cyc_A)*msk_A)
        cyc_loss_B = tf.reduce_mean(tf.abs(self.input_B - self.cyc_B)*msk_B)
        p2p_loss_A = tf.reduce_mean(tf.abs(self.input_A - self.fake_A)*msk_A)
        p2p_loss_B = tf.reduce_mean(tf.abs(self.input_B - self.fake_B)*msk_B)
        disc_loss_A = tf.reduce_mean(tf.abs(self.fake_rec_A - 1))
        disc_loss_B = tf.reduce_mean(tf.abs(self.fake_rec_B - 1))

        lossmap = {'dis': (disc_loss_A, disc_loss_B), 'p2p': (p2p_loss_A, p2p_loss_B), 'cls': (cls_loss_A, cls_loss_B), 'cyc': (cyc_loss_B, cyc_loss_A)}
        g_loss_A = 0; g_loss_B = 0
        for tsk in tasks:
            print(tsk)
            g_loss_A = g_loss_A + lossmap[tsk][1]
            g_loss_B = g_loss_B + lossmap[tsk][0]

        d_loss_A = (tf.reduce_mean(tf.abs(self.fake_pool_rec_A)) + tf.reduce_mean(tf.abs(self.rec_A-1))) / 2.0
        d_loss_B = (tf.reduce_mean(tf.abs(self.fake_pool_rec_B)) + tf.reduce_mean(tf.abs(self.rec_B-1))) / 2.0

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)


    def save_training_images(self, sess, epoch):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for ptr in range(0, max_images):
            inputA, inputB, label = self.inputAB(self.imdb_train[ptr], cycload=True, augment=False)
            if (inputA is not None) & (inputB is not None):
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                        [self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                        feed_dict={self.input_A: inputA[0:1,:,:,:], self.input_B: inputB[0:1,:,:,:]})
                sio.savemat(output_path+"/fake_" + str(epoch) + "_" + str(ptr) + ".mat",
                            {'fake_A': fake_A_temp[0], 'fake_B': fake_B_temp[0],
                             'cyc_A': cyc_A_temp[0], 'cyc_B': cyc_B_temp[0],
                             'input_A': inputA[0], 'input_B': inputB[0]})
                break

    def matrics_calc(self, testvals, labels, pos=0, neg=1):
        mean = np.mean(testvals, axis=0)
        # testvals = testvals - mean
        print(np.sum(testvals, axis=0))
        AUC = metrics.roc_auc_score(y_score=np.transpose(testvals), y_true=np.transpose(labels), average='samples')
        TP = 0; TN=0; FP=0; FN=0
        f = 1.00
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

                if (save_training_images):
                    self.save_training_images(sess, epoch)
                trainlabels_A = []; trainprobs_A = []; losses_A = 0
                trainlabels_B = []; trainprobs_B = []; losses_B = 0

                for ptr in range(0, min(max_images, len(self.imdb_train))):
                    print("In the iteration ", ptr, self.imdb_train[ptr])
                    inputA, inputB, label = self.inputAB(self.imdb_train[ptr], cycload=True, augment=True)

                    if inputA is not None and (epoch < 30):
                        _, prob, loss = sess.run([self.cls_A_trainer, self.prob_A, self.cls_loss_A],
                                                     feed_dict={self.input_A: inputA, self.label_holder: label})
                        losses_A = losses_A + loss; trainlabels_A.append(label); trainprobs_A.append(prob)

                    if inputB is not None and (epoch < 50):
                        _, prob, loss = sess.run([self.cls_B_trainer, self.prob_B, self.cls_loss_B],
                                                     feed_dict={self.input_B: inputB, self.label_holder: label})
                        losses_B = losses_B + loss; trainlabels_B.append(label); trainprobs_B.append(prob)

                    if (inputA is not None) and (inputB is not None):
                        _, fake_B_temp = sess.run([self.g_A_trainer, self.fake_B],
                                                    feed_dict={self.input_A: inputA, self.input_B: inputB, self.lr: curr_lr})
                        if 'dis' in tasks:
                            sess.run([self.d_B_trainer],
                                     feed_dict={self.input_A: inputA, self.input_B: inputB, self.lr: curr_lr,
                                                    self.fake_pool_B: fake_B_temp})

                        _, fake_A_temp = sess.run([self.g_B_trainer, self.fake_A],
                                                    feed_dict={self.input_A: inputA, self.input_B: inputB, self.lr: curr_lr})
                        if 'dis' in tasks:
                            sess.run(self.d_A_trainer,
                                     feed_dict={self.input_A: inputA, self.input_B: inputB, self.lr: curr_lr,
                                                    self.fake_pool_A: fake_A_temp})

                if (epoch < 30):
                    print('loss_A:', losses_A / len(trainprobs_A), self.matrics_calc(np.concatenate(trainprobs_A), np.concatenate(trainlabels_A), pos=1, neg=0))
                if (epoch < 50):
                    print('loss_B:', losses_B / len(trainprobs_B), self.matrics_calc(np.concatenate(trainprobs_B), np.concatenate(trainlabels_B), pos=1, neg=0))

                sess.run(tf.assign(self.global_step, epoch + 1))


    def test(self):
        ''' Testing Function'''
        print("Testing the results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(init)
            print(chkpt_fname)
            saver.restore(sess, chkpt_fname)
            if not os.path.exists(outputA_path):
                os.makedirs(outputA_path)
            if not os.path.exists(outputB_path):
                os.makedirs(outputB_path)
            img_out = np.zeros([181, 217, 181], np.uint8)
            for ptr in range(min(len(self.imdb_test), max_images)):
                inputA, inputB, label = self.inputAB(self.imdb_test[ptr], cycload=False, augment=False)
                if inputA is None: continue
                filename = self.imdb_test[ptr][0]
                fake_B_temp = sess.run(self.fake_B, feed_dict={self.input_A: inputA})
                img_out[18:162, 22:198, 10:154] = ((np.squeeze(fake_B_temp) + 1) * 128).astype(np.uint8)
                sio.savemat(outputA_path + filename + '.mat', {'IMG': img_out})
                print(filename)

            for ptr in range(min(len(self.imdb_test), max_images)):
                inputA, inputB, label = self.inputAB(self.imdb_test[ptr], cycload=False, augment=False)
                if inputB is None: continue
                filename = self.imdb_test[ptr][0]
                fake_A_temp = sess.run(self.fake_A, feed_dict={self.input_B: inputB})
                img_out[18:162, 22:198, 10:154] = ((np.squeeze(fake_A_temp) + 1) * 96).astype(np.uint8)
                sio.savemat(outputB_path + filename + '.mat', {'IMG': img_out})
                print(filename)

    def eval(self):
        print("eval the synthetic results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(init)
            print(chkpt_fname)
            saver.restore(sess, chkpt_fname)
            if not os.path.exists(outputA_path):
                os.makedirs(outputA_path)
            if not os.path.exists(os.path.join(output_path, str(tasks), 'samples')):
                os.makedirs(os.path.join(output_path, str(tasks), 'samples'))
            MAE = []; SSIM = []; PSNR = []
            for ptr in range(min(len(self.imdb_test), max_images)):
                inputA, inputB, label = self.inputAB(self.imdb_test[ptr], cycload=False, augment=False)
                if (inputB is None) | (inputA is None): continue
                filename = self.imdb_test[ptr][0]
                print(filename)
                fake_A, fake_B = sess.run([self.fake_A, self.fake_B], feed_dict={self.input_A: inputA[0:1], self.input_B: inputB[0:1]})
                print([np.mean(np.abs(fake_A-inputA)), np.mean(np.abs(fake_B-inputB))])
                MAE.append([np.mean(np.abs(fake_A-inputA)), np.mean(np.abs(fake_B-inputB))])
                SSIM.append([ssim(inputA[0], fake_A[0], multichannel=True), ssim(inputB[0], fake_B[0], multichannel=True)])
                PSNR.append([psnr(inputA[0]/2, fake_A[0]/2), psnr(inputB[0]/2, fake_B[0]/2)])

                imsave(os.path.join(output_path, str(tasks), 'samples', filename + '.bmp'),
                       np.concatenate((np.array((inputA[:, :, :, 72, :]) * 128 + 128).astype(np.uint8).reshape([img_width, img_height]),
                                       np.array((inputB[:, :, :, 72, :]) * 96 + 96).astype(np.uint8).reshape([img_width, img_height]),
                                       np.array((fake_A[:, :, :, 72, :]) * 128 + 128).astype(np.uint8).reshape([img_width, img_height]),
                                       np.array((fake_B[:, :, :, 72, :]) * 96 + 96).astype(np.uint8).reshape([img_width, img_height])), axis=1), 'bmp')
            print(np.mean(MAE, axis=0), np.mean(SSIM, axis=0), np.mean(PSNR, axis=0))
            print(np.std(MAE, axis=0), np.std(SSIM, axis=0), np.std(PSNR, axis=0))

    def extra(self):
        print("eval the classification results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(100, 101, 1):
                # chkpt_fname = tf.train.latest_checkpoint(check_dir)
                print("epoch-{0}".format(epoch))
                chkpt_fname = check_dir + str(tasks)+'-{0}'.format(epoch)
                saver.restore(sess, chkpt_fname)
                testfeats_A = []; testlabels_A = []; testprobs_A = []; losses_A = 0
                testfeats_B = []; testlabels_B = []; testprobs_B = []; losses_B = 0
                reader = tf.train.NewCheckpointReader(chkpt_fname)
                wa = reader.get_tensor('CLS/cls_A/fc_layer/w')
                wb = reader.get_tensor('CLS/cls_B/fc_layer/w')
                rwa = np.reshape(wa[:, 0], [64, -1])
                rwb = np.reshape(wb[:, 0], [64, -1])

                # np.savetxt('cls_A.txt', np.sum(rwa*rwa, axis=0), '%.10f')
                # np.savetxt('cls_B.txt', np.sum(rwb*rwb, axis=0), '%.10f')

                for ptr in range(min(len(self.imdb_test), max_images)):
                    inputA, inputB, label = self.inputAB(self.imdb_test[ptr], cycload=True, augment=False)
                    if inputA is not None:
                        prob, loss, feats = sess.run([self.prob_A, self.cls_loss_A, self.feats_A], feed_dict={self.input_A: inputA, self.label_holder: label})
                        losses_A = losses_A + loss; testlabels_A.append(label); testprobs_A.append(prob), testfeats_A.append(feats[-1])
                    if inputB is not None:
                        prob, loss, feats = sess.run([self.prob_B, self.cls_loss_B, self.feats_B], feed_dict={self.input_B: inputB, self.label_holder: label})
                        losses_B = losses_B + loss; testlabels_B.append(label); testprobs_B.append(prob), testfeats_B.append(feats[-1])
                print('loss_A:', losses_A / len(testprobs_A),
                      self.matrics_calc(np.concatenate(testprobs_A), np.concatenate(testlabels_A), pos=1, neg=0))
                print('loss_B:', losses_B / len(testprobs_B),
                      self.matrics_calc(np.concatenate(testprobs_B), np.concatenate(testlabels_B), pos=1, neg=0))

    def cross_extra(self):
        print("eval the synthetic classification results")
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(100, 101, 1):
                print("epoch-{0}".format(epoch))
                chkpt_fname = check_dir + str(tasks) + '-{0}'.format(epoch)
                saver.restore(sess, chkpt_fname)
                testfeats_A = []; testlabels_A = []; testprobs_A = []; losses_A = 0
                testfeats_B = []; testlabels_B = []; testprobs_B = []; losses_B = 0

                for ptr in range(min(len(self.imdb_test), max_images)):
                    inputA, inputB, label = self.inputAB(self.imdb_test[ptr], cycload=True, augment=False)
                    if inputB is not None:
                        fake_A = sess.run(self.fake_A, feed_dict={self.input_B: inputB[0:1]})
                        prob, loss, feats = sess.run([self.prob_A, self.cls_loss_A, self.feats_A],
                                                     feed_dict={self.input_A: fake_A, self.label_holder: label})
                        losses_A = losses_A + loss; testlabels_A.append(label)
                        testprobs_A.append(prob)
                    if inputA is not None:
                        fake_B = sess.run(self.fake_B, feed_dict={self.input_A: inputA[0:1]})
                        prob, loss, feats = sess.run([self.prob_B, self.cls_loss_B, self.feats_B],
                                                     feed_dict={self.input_B: fake_B, self.label_holder: label})
                        losses_B = losses_B + loss; testlabels_B.append(label)
                        testprobs_B.append(prob)
                print('loss_A:', losses_A / len(testprobs_A),
                      self.matrics_calc(np.concatenate(testprobs_A), np.concatenate(testlabels_A), pos=1, neg=0))
                print('loss_B:', losses_B / len(testprobs_B),
                      self.matrics_calc(np.concatenate(testprobs_B), np.concatenate(testlabels_B), pos=1, neg=0))


def main():
    model = CCGAN()
    model.input_setup_adni(input_path)
    model.model_setup()
    model.loss_calc()
    if model_stats == 'train':
        model.train()
    elif model_stats == 'test':
        model.test()
    elif model_stats == 'extra':
        model.extra()
    elif model_stats == 'eval':
        model.eval()
    else:
        model.cross_extra()


if __name__ == '__main__':
    main()