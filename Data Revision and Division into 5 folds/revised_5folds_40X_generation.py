#!/usr/bin/env python
#coding:utf-8

""" This script creates the 5 folds used in the experiments reported in [1].

    It ASSUMES:
        - that the path is relative to the current directory.
        - that BreaKHis_v1.tar.gz was decompressed in current directory.

    It REQUIRES:
        - text files dsfold1.txt, ... dsfold5.txt located in current directory.

    ****Approximately 20 GB of disk space will be allocated for fold1,... fold5 directories.


    -------
    [1] Spanhol, F.A.; Oliveira, L.S.; Petitjean, C.; Heutte, L. "A Dataset for Breast Cancer Histopathological Image Classification". Biomedical Engineering, IEEE Transactions on. Year: 2015, DOI: 10.1109/TBME.2015.2496264

"""
__author__ = "Fabio Alexandre Spanhol"
__email__ = "faspanhol@gmail.com"


# import sys
import os
import shutil
# from glob import glob
# import time


# -----------------------------------------------------------------------------
def create_folds_from_ds(dst_path='./dataset', folds=(1,5), target_magnification = '40'):
    """Creates a structure of directories containing images
        selected from BreaKHis_v1 dataset.
    """
    root_dir = './BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}

    for nfold in folds:
        print('fold:', nfold)
        # directory for nth-fold
        dst_dir = dst_path + '/revised_fold%s' % nfold + '_' + target_magnification + 'x'
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        dst_dir = dst_dir + '/%s'

        # image list
        db = open('revised_fold_%s.txt' % nfold)

        for row in db.readlines():
            columns = row.split('|')
            imgname = columns[0]
            mag = columns[1]  # 40, 100, 200, or 400
            clstype = columns[2]  # B or M or N
            grp = columns[3].strip()  # train or test

            if mag == target_magnification:
                dst_subdir0 = dst_dir % grp + '/'
                if not os.path.exists(dst_subdir0):
                    os.mkdir(dst_subdir0)

                dst_subdir0 = dst_subdir0 + '/%s'
                dst_subdir1 = dst_subdir0 % clstype + '_' + grp + '/'
                if not os.path.exists(dst_subdir1):
                    os.mkdir(dst_subdir1)

                tumor = imgname.split('-')[0].split('_')[-1]
                srcfile = srcfiles[tumor]

                s = imgname.split('-')
                sub = s[0] + '_' + s[1] + '-' + s[2]

                srcfile = srcfile % (root_dir, sub, mag, imgname)
                
                #B_M = imgname.split('-')[0].split('_')[1]
                if clstype == 'B':
                    imgname2 = '0_' + imgname
                elif clstype == 'M':
                    imgname2 = '1_' + imgname
                else:
                    imgname2 = '2_' + imgname
                dstfile = dst_subdir1 + imgname2

                #print "Copying from [%s] to [%s]" % (srcfile, dstfile)
                shutil.copy(srcfile, dstfile)
                #time.sleep(1)
        #print '\n\n\t\tFold #%d finished.\n' % nfold
    db.close()
    #print "\nProcess completed."
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    create_folds_from_ds()