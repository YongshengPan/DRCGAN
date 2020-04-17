import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
from scipy import io as sio
import os
import shutil
from PIL import Image
import random
import time
import sys
from sklearn import svm
from sklearn import metrics
import csv
from layers import *
from modelsingle import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


to_train_or_test = 'test'
to_restore = False
positive = 1
negative = 2

img_width = 181
img_height = 217
img_depth = 181
img_layer = 1

task = 'ADNIO'
output_path = "./outputsingle/"
outputA_path = output_path + "/train/fakeA/"
outputB_path = output_path + "/train/fakeB/"
input_path = "input/" + task + "/"
id_check = '-40'
backbone = 'fc_vgg3d'
modality = "PET"
check_dir = "./outputsingle/" + task + "/" + modality+"/"
ckpt_fname = check_dir + backbone + id_check
max_epoch = 41
max_images = 2000
step = 1
numldmk = 1
batch_size = 1
sample_size = 27
groups = {'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 1, 'sSCD': 0, 'pSCD': 1, 'sSMC': 0, 'pSMC': 1, 'MCI': 1, 'SMC': 6, 'Autism': 1, 'Control': 0}

class CycleGAN():

    def inputAB(self, imdb, cycload=True, augment=True, rngldmk=None):
        if rngldmk is None:
            rngldmk = [self.index_landmark]
        flnm, grp = imdb
        if flnm in self.datapool:
            image, label, ldmk = self.datapool[flnm]
        else:
            label = np.zeros(2, np.float32)
            cls = groups[grp]
            if cls in [0, 1]: label[cls] = 1

            mfile = modality + '/' + flnm + '.mat'
            # pfile = 'PET/' + flnm + '.mat'
            lfile = 'landmark.mat'

            if os.path.exists(input_path + mfile):
                ldmk = sio.loadmat(input_path + lfile)['landmark']
                data = sio.loadmat(input_path + mfile)
                image = np.array(data['IMG'])
            else:
                image = None
                ldmk = None
                # print(mfile)

            if cycload:
                self.datapool[flnm] = image, label, ldmk

        img = []
        labels = []

        if image is None:
            img = image
        elif augment:
            image = image.astype(np.float32) - 128
            for idx in rngldmk:
                patches=[]
                for dx in range(-step, step+1):
                    for dy in range(-step, step+1):
                        for dz in range(-step, step+1):
                            ld = np.round(ldmk[:, idx - 1]).astype(np.int)
                            ld[0] = ld[0] + dx
                            ld[1] = ld[1] + dy
                            ld[2] = ld[2] + dz
                            patches.append(image[(ld[0] - 16):(ld[0] + 16), (ld[1] - 16):(ld[1] + 16), (ld[2] - 16):(ld[2] + 16), np.newaxis])
                img.append(patches)
            # img = np.concatenate(img)
            for idy in range((1+2*step)**3):
                labels.append(label)
        else:
            image = image.astype(np.float32) - 128
            for idx in rngldmk:
                patches = []
                ld = np.round(ldmk[:, idx - 1]).astype(np.int)
                patches.append(image[(ld[0] - 16):(ld[0] + 16), (ld[1] - 16):(ld[1] + 16), (ld[2] - 16):(ld[2] + 16), np.newaxis])
                img.append(patches)
            labels.append(label)
            # img = np.concatenate(img)
        return img, labels


    def get_database(self, imdbname, vldgrp=None):
        imdb = []
        with open(imdbname, newline='') as csvfile:
            imdbreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in imdbreader:
                if (vldgrp is None):
                    imdb.append(row[1:3])
                elif (row[2] in vldgrp):
                    imdb.append(row[1:3])
        return imdb

    def input_setup_aible(self, input_path):
        self.datapool = {}
        self.imdb_train = self.get_database(input_path + '/ABIDE_imdb.csv', vldgrp=['Autism', 'Control'])
        self.imdb_test = self.get_database(input_path + '/ABIDE_imdb.csv', vldgrp=['Autism', 'Control'])
        self.imdb_train = self.imdb_train[1:-1: 2]
        self.imdb_test = self.imdb_test[0:-1: 2]
        print(len(self.imdb_train))
        print(len(self.imdb_test))

    def input_setup_adni(self, input_path):
        self.datapool = {}
        self.imdb_train = self.get_database(input_path + '/ADNI1_imdb_18m.csv', vldgrp=["AD", "CN"])
        # self.imdb_test  = self.get_database(input_path + '/ADNI2_imdb_18m.csv', vldgrp=["sMCI", "CN"])
        self.imdb_test  = self.get_database(input_path + '/ADNI2_imdb_18m.csv', vldgrp=["AD", "CN"])
        # self.imdb_test = self.get_database(input_path + '/sSCDpSCDwHC.csv', vldgrp=["pSCD", "sSCD"])
        # self.imdb_train = self.imdb_train[2:-1:2]
        # self.imdb_test = self.imdb_test[1:-1:2]
        print(len(self.imdb_train))
        print(len(self.imdb_test))

    def model_setup(self):

        self.input_holder = tf.placeholder(tf.float32, [numldmk, None, 32, 32, 32, img_layer], name="input_A")
        self.label_holder = tf.placeholder(tf.float32, [None, 2], name="label_A")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:
            # self.logits, self.prob, self.fc_in, self.feats = bulid_fv_resNet(self.input, 2, training=True, usBN=True)
            self.logit, self.prob, self.feats = build_multinstance(self.input_holder, 16, numldmk, backbone=backbone,
                                                                   #means=self.means_holder, vars=self.vars_holder,
                                                                   name="model")
            print(self.logit, self.prob, self.feats)

        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit, labels=self.label_holder))

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-8)
        # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.9)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        self.trainer = optimizer.minimize(self.loss)


    def matrics_calc(self, testvals, labels, pos=3, neg=2):
        AUC = metrics.roc_auc_score(y_score=np.transpose(testvals), y_true=np.transpose(labels), average='samples')
        TP = 0; TN=0; FP=0; FN=0
        print(np.shape(testvals), np.shape(labels))
        for idx in range(len(testvals)):
            if (np.argmax(labels[idx], axis=0) == pos) & (np.argmax(testvals[idx], axis=0) == pos):
                TP = TP + 1
            if (np.argmax(labels[idx], axis=0) == neg) & (np.argmax(testvals[idx], axis=0) == neg):
                TN = TN + 1
            if (np.argmax(labels[idx], axis=0) == neg) & (np.argmax(testvals[idx], axis=0) == pos):
                FP = FP + 1
            if (np.argmax(labels[idx], axis=0) == pos) & (np.argmax(testvals[idx], axis=0) == neg):
                FN = FN + 1
        print(TP, TN ,FP, FN)
        ACC = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        SEN = (TP) / (TP + FN + 1e-6)
        SPE = (TN) / (TN + FP + 1e-6)
        PPV = (TP) / (TP + FP + 1e-6)
        F_score = (2 * SEN * PPV) / (SEN + PPV + 1e-6)
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)+ 1e-6)
        return [AUC, ACC, SEN, SPE, F_score, MCC]

    def train(self, ind_ldmk=40):
        ''' Training Function '''
        self.index_landmark = ind_ldmk
        if task in ['ADNI', 'ADNIO', 'ADNIP', 'ADNIS']:
            self.input_setup_adni(input_path)
        elif task == 'AIBLE':
            self.input_setup_aible(input_path)
        self.model_setup()  # Build the network
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())# Initializing the variables
        saver = tf.train.Saver(max_to_keep=0)
        # saver1 = tf.train.Saver(var_list = restore_vars)
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if to_restore:
                chkpt_fname = check_dir + backbone + '{0}'.format(self.index_landmark) + id_check
                # chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step), max_epoch):
                print("In the epoch ", epoch)
                saver.save(sess, os.path.join(check_dir, backbone + '{0}'.format(self.index_landmark)), global_step=epoch)

                trainlabels = []; trainvals = []; trainfcs = []; trainfeats = []; loss_All=0
                for ptr in range(0, min(max_images, len(self.imdb_train))):
                    inputA, labelA = self.inputAB(self.imdb_train[ptr], cycload=True, augment=True)
                    if inputA is None: continue
                    for idx in range(1):
                        _, logit, prob, feats, loss = sess.run([self.trainer, self.logit, self.prob, self.feats, self.loss],
                                                           feed_dict={self.input_holder: inputA, self.label_holder: labelA})
                        loss_All = loss_All + loss
                        trainlabels.append(np.mean(labelA, axis=0))
                        trainvals.append(np.mean(prob, axis=0))
                        trainfeats.append(np.mean(feats, axis=(0,1,2,3)))
                        # print(np.mean(logits, axis=0), np.mean(prob, axis=0), loss, np.mean(labelA, axis=0))
                print('loss_A:', loss_All/len(trainvals), self.matrics_calc(trainvals, trainlabels, pos=positive - 1, neg=negative - 1))
                if epoch % 10 == 0:
                    testlabels = []; testvals = []; testfcs = []; testfeats = []
                    for ptr in range(min(len(self.imdb_test), max_images)):
                        inputA, labelA = self.inputAB(self.imdb_test[ptr], cycload=True, augment=False)
                        if inputA is None: continue
                        logits, prob, feats, loss = sess.run([self.logit, self.prob, self.feats, self.loss],
                                                           feed_dict={self.input_holder: inputA, self.label_holder: labelA})
                        testlabels.append(np.mean(labelA, axis=0))
                        testvals.append(np.mean(prob, axis=0))
                        testfeats.append(np.mean(feats, axis=(0,1,2,3)))
                    print(self.matrics_calc(testvals, testlabels, pos=positive - 1, neg=negative - 1))
                    # self.svmtest(trainfeats, trainlabels, testfeats, testlabels)
                sess.run(tf.assign(self.global_step, epoch + 1))
        tf.reset_default_graph()

    def test(self, ind_ldmk = 40):
        ''' Testing Function'''
        print("Testing the results")
        self.index_landmark = ind_ldmk
        self.input_setup_adni(input_path)
        self.model_setup()
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        config.gpu_options.allow_growth = True
        chkpt_fname = check_dir + backbone+'{0}'.format(self.index_landmark) + id_check
        with tf.Session(config=config) as sess:
            sess.run(init)
            saver.restore(sess, chkpt_fname)
            if not os.path.exists(outputA_path):
                os.makedirs(outputA_path)
            if not os.path.exists(outputB_path):
                os.makedirs(outputB_path)

            trainvals = []; trainlabels = []; trainfcs = []; trainfeats = []; loss_All=0
            for ptr in range(0, min(max_images, len(self.imdb_train))):
                inputA, labelA = self.inputAB(self.imdb_train[ptr], cycload=True, augment=False)
                if inputA is None: continue
                for idx in range(1):
                    logits, prob, feats, loss = sess.run(
                        [self.logit, self.prob, self.feats, self.loss],
                        feed_dict={self.input_holder: inputA, self.label_holder: labelA})
                    loss_All = loss_All + loss
                    trainlabels.append(np.mean(labelA, axis=0))
                    trainvals.append(np.mean(prob, axis=0))
                    trainfeats.append(feats)
            print('loss_A:', loss_All / len(trainvals),
                  self.matrics_calc(trainvals, trainlabels, pos=positive - 1, neg=negative - 1))

            testlabels = []; testvals = []; testfcs = []; testfeats = []; loss_All=0
            for ptr in range(min(len(self.imdb_test), max_images)):
                inputA, labelA = self.inputAB(self.imdb_test[ptr], cycload=True, augment=False)
                if inputA is None: continue
                logits, prob, feats, loss = sess.run([self.logit, self.prob, self.feats, self.loss],
                                                     feed_dict={self.input_holder: inputA, self.label_holder: labelA})
                loss_All = loss_All + loss
                testlabels.append(np.mean(labelA, axis=0))
                testvals.append(np.mean(prob, axis=0))
                testfeats.append(feats)
            print(self.matrics_calc(testvals, testlabels, pos=positive - 1, neg=negative - 1))

            # sio.savemat(outputA_path + 'output{0}.mat'.format(self.index_landmark),
            #                 {'trainfeats': trainfeats, 'trainfcs': trainfcs, 'trainvals': trainvals, 'trainlabels': trainlabels,
            #                  'testfeats': testfeats, 'testfcs': testfcs, 'testvals': testvals, 'testlabels': testlabels})
            # self.svmtest(trainfeats, trainlabels, testfeats, testlabels)
        tf.reset_default_graph()


    def svmtest(self, traindata, trainlabel, testdata, testlabel):
        print("Classification by SVM")
        # print(np.shape(traindata))
        # print(trainlabel)
        trainlabel = np.transpose(trainlabel)[positive]
        testlabel = np.transpose(testlabel)[positive]
        clf =svm.LinearSVC()
        clf.fit(traindata, trainlabel)
        decision = clf.predict(testdata)
        print(np.mean(decision == testlabel))

def main():

    if to_train_or_test == 'train':
        # for ldid in [29, 37, 38, 39, 40, 41, 42]:
        for ldid in range(1, 117):#:[55, 56, 73, 74, 75, 76, 83]
        # for ldid in [85, 87, 89, 90, 95, 96, 97, 98]:  # range(1, 117):
            print('landmark={0}'.format(ldid))
            model = CycleGAN()
            model.train(ind_ldmk=ldid)
    else:
        model = CycleGAN()
        for ldid in range(1,117):
            print('landmark={0}'.format(ldid))
            model.test(ind_ldmk=ldid)


if __name__ == '__main__':
    main()