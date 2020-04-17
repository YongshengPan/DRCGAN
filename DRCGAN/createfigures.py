import numpy as np
from scipy import misc as smics
from matplotlib import pyplot
import os
from PIL import Image, ImageDraw
from skimage.color import gray2rgb
import random
import time
import sys
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

selected_feat = 0, 1, 2, 3, 4,
tasks = 'cyc', 'dis',
model_stats = 'eval'

groups = {'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 0, 'sSCD': 0, 'pSCD': 1, 'MCI': 1, 'sSMC': 0, 'pSMC': 1, 'SMC': 0,
          'sCN': 0, 'pCN': 0, 'ppCN': 1, 'Autism': 1, 'Control': 0}


output_path = "./output_{0}_fm{1}/ADNIO/{0}/samples/"
outputA_path = output_path + "{0}/PET/".format(tasks)
outputB_path = output_path + "{0}/MRI/".format(tasks)
input_path = "D:/ADNI/CycleGAN/" + "input/ADNIO/"

max_images = 2000

img_width = 144
img_height = 176
img_depth = 144
img_layer = 1


class CCGAN():

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
        # self.imdb_test = self.get_database(input_path + '/ADNI2_imdb_36m.csv', ['pMCI', 'sMCI'])
        print(len(self.imdb_train))
        print(len(self.imdb_test))


    def eval(self):
        print("eval the synthetic results")
        task_GAN = 'dis',
        task_cycGAN = 'cyc', 'dis',
        task_conGAN = 'dis', 'p2p',
        task_FCGAN = 'cls', 'dis',
        task_FCcycGAN = 'cls', 'cyc', 'dis',
        task_FCconGAN = 'cls', 'dis', 'p2p',
        output_path_GAN = output_path.format(task_GAN, selected_feat)
        output_path_cycGAN = output_path.format(task_cycGAN, selected_feat)
        output_path_conGAN = output_path.format(task_conGAN, selected_feat)
        output_path_FCGAN = output_path.format(task_FCGAN, selected_feat)
        output_path_FCcycGAN = output_path.format(task_FCcycGAN, selected_feat)
        output_path_FCconGAN = output_path.format(task_FCconGAN, selected_feat)
        output_path_syn = [output_path_GAN, output_path_cycGAN, output_path_conGAN, output_path_FCGAN, output_path_FCcycGAN, output_path_FCconGAN]

        mri_rgb = [255, 0, 255]; pet_rgb = [0, 255, 0]
        for ptr in range(min(len(self.imdb_test), max_images)):
            filename = self.imdb_test[ptr][0]
            print(filename)
            IMG = []
            image = np.ones((352, 352))*255
            for ipath in output_path_syn:
                fl = ipath + filename + '.bmp'
                if os.path.exists(fl):
                    img = np.flip(np.transpose(smics.imread(fl)[:, 352:704]), axis=0)
                    image[:, 0:144] = img
                    image[  0:144, 168:312] = smics.imresize(img[ 94:166, 62:134], 2.0)
                    image[208:352, 168:312] = smics.imresize(img[270:342, 62:134], 2.0)
                    rgb_img = gray2rgb(image)
                    rgb_img[94:166, (62, 133), :] = pet_rgb
                    rgb_img[270:342, (62, 133), :] = mri_rgb
                    rgb_img[(94, 165), 62:134, :] = pet_rgb
                    rgb_img[(270, 341), 62:134, :] = mri_rgb
                    rgb_img[(0, 143), 168:312] = pet_rgb
                    rgb_img[(208, 351), 168:312] = mri_rgb
                    rgb_img[0:144, (168, 312)] = pet_rgb
                    rgb_img[208:352, (168, 312)] = mri_rgb
                    rgb_img[112, 134:168] = pet_rgb
                    rgb_img[288, 134:168] = mri_rgb
                    smics.imsave(fl.replace('.bmp', '_N.bmp'), rgb_img)
                    IMG.append(rgb_img)

                    img = np.flip(np.transpose(smics.imread(fl)[:, 0:352]), axis=0)
                    image[:, 0:144] = img
                    image[0:144, 168:312] = smics.imresize(img[94:166, 62:134], 2.0)
                    image[208:352, 168:312] = smics.imresize(img[270:342, 62:134], 2.0)
                    rgb_img = gray2rgb(image)
                    rgb_img[94:166, (62, 133), :] = pet_rgb
                    rgb_img[270:342, (62, 133), :] = mri_rgb
                    rgb_img[(94, 165), 62:134, :] = pet_rgb
                    rgb_img[(270, 341), 62:134, :] = mri_rgb
                    rgb_img[(0, 143), 168:312] = pet_rgb
                    rgb_img[(208, 351), 168:312] = mri_rgb
                    rgb_img[0:144, (168, 312)] = pet_rgb
                    rgb_img[208:352, (168, 312)] = mri_rgb
                    rgb_img[112, 134:168] = pet_rgb
                    rgb_img[288, 134:168] = mri_rgb
                    smics.imsave(fl.replace('.bmp', '_T.bmp'), rgb_img)
                    if ipath == output_path_FCconGAN:
                        IMG.append(rgb_img)

            if len(IMG) > 6:
                IMG = np.concatenate(IMG, axis=1)
                smics.imsave(output_path_FCconGAN + filename + 'NT.bmp', IMG)

def main():
    model = CCGAN()
    model.input_setup_adni(input_path)
    model.eval()


if __name__ == '__main__':
    main()