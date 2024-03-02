import numpy as np
import cv2
import math
def color_deconvolution(image_input):
    
    #image_input = cv2.imread('G1.2_modified.jpg')
    ImgB, ImgG, ImgR = cv2.split(image_input)
    StainingVectorID=2;
    DyeToBeRemovedID=0;
    doIcross=1;
    # %% INTERNAL PARAMETERS:
        
    [rowR, colR]  = np.shape(ImgR)
    #nchR=1
    Dye01_transmittance = np.zeros([rowR, colR]);
    Dye02_transmittance = np.zeros([rowR, colR]);
    Dye03_transmittance = np.zeros([rowR, colR]);
    ImgR_back           = np.zeros([rowR, colR]);
    ImgG_back           = np.zeros([rowR, colR]);
    ImgB_back          	= np.zeros([rowR, colR]);
    Const1ForNoLog0     = 0;
    #Const2ForNoLog0     = 0;
    # %% COEFFICIENTS CONNECTED TO THE StainingVectorID:
        
    [SMrow, SMcol, SMnch]  = [np.size(StainingVectorID), 1, 1];
    if (SMrow==3 and SMcol==3 and SMnch==1):
        SM = StainingVectorID;
        StainingVectorID = 21;
    elif (SMrow==1 and SMcol==1 and SMnch==1):
        if (StainingVectorID<1 or StainingVectorID>21):
            print('ERROR: StainingVectorID is not correctly defined!');
       
    if StainingVectorID==1:
        # "H&E"
        MODx = [0.644211, 0.092789, 0];
        MODy = [0.716556, 0.954111, 0];
        MODz = [0.266844, 0.283111, 0];
    elif StainingVectorID==2:
        # "H&E 2"
        MODx = [0.49015734, 0.04615336, 0];
        MODy = [0.76897085, 0.8420684, 0];
        MODz = [0.41040173, 0.5373925, 0];
    elif StainingVectorID==3:
        # "H DAB"
        MODx = [0.650, 0.268, 0];
        MODy = [0.704, 0.570, 0];
        MODz = [0.286, 0.776, 0];
    elif StainingVectorID==4:
        # "H&E DAB"
        MODx = [0.650, 0.072, 0.268];
        MODy = [0.704, 0.990, 0.570];
        MODz = [0.286, 0.105, 0.776];
    elif StainingVectorID==5:
        # "NBT/BCIP Red Counterstain II"
        MODx = [0.62302786, 0.073615186, 0.7369498];
        MODy = [0.697869, 0.79345673, 0.0010];
        MODz = [0.3532918, 0.6041582, 0.6759475];
    elif StainingVectorID==6:
        # "H DAB NewFuchsin"
        MODx = [0.5625407925, 0.26503363, 0.0777851125];
        MODy = [0.70450559, 0.68898016, 0.804293475];
        MODz = [0.4308375625, 0.674584, 0.5886050475];
    elif StainingVectorID==7:
        # "H HRP-Green NewFuchsin"
        MODx = [0.8098939567, 0.0777851125, 0.0];
        MODy = [0.4488181033, 0.804293475, 0.0];
        MODz = [0.3714423567, 0.5886050475, 0.0];
    elif StainingVectorID==8:
        # "Feulgen LightGreen"
        MODx = [0.46420921, 0.94705542, 0.0];
        MODy = [0.83008335, 0.25373821, 0.0];
        MODz = [0.30827187, 0.19650764, 0.0];
    elif StainingVectorID==9:
        # "Giemsa"
        MODx = [0.834750233, 0.092789, 0.0];
        MODy = [0.513556283, 0.954111, 0.0];
        MODz = [0.196330403, 0.283111, 0.0];
    elif StainingVectorID==10:
        # "FastRed FastBlue DAB"
        MODx = [0.21393921, 0.74890292, 0.268];
        MODy = [0.85112669, 0.60624161, 0.570];
        MODz = [0.47794022, 0.26731082, 0.776];
    elif StainingVectorID==11:
        # "Methyl Green DAB"
        MODx = [0.98003, 0.268, 0.0];
        MODy = [0.144316, 0.570, 0.0];
        MODz = [0.133146, 0.776, 0.0];
    elif StainingVectorID==12:
        # "H AEC"
        MODx = [0.650, 0.2743, 0.0];
        MODy = [0.704, 0.6796, 0.0];
        MODz = [0.286, 0.6803, 0.0];
    elif StainingVectorID==13:
        # "Azan-Mallory"
        MODx = [0.853033, 0.09289875, 0.10732849];
        MODy = [0.508733, 0.8662008, 0.36765403];
        MODz = [0.112656, 0.49098468, 0.9237484];
    elif StainingVectorID==14:
        # "Masson Trichrome"
        MODx = [0.7995107, 0.09997159, 0.0];
        MODy = [0.5913521, 0.73738605, 0.0];
        MODz = [0.10528667, 0.6680326, 0.0];
    elif StainingVectorID==15:
        # "Alcian blue & H"
        MODx = [0.874622, 0.552556, 0.0];
        MODy = [0.457711, 0.7544, 0.0];
        MODz = [0.158256, 0.353744, 0.0];
    elif StainingVectorID==16:
        # "H PAS"
        MODx = [0.644211, 0.175411, 0.0];
        MODy = [0.716556, 0.972178, 0.0];
        MODz = [0.266844, 0.154589, 0.0];
    elif StainingVectorID==17:
        # "Brilliant_Blue"
        MODx = [0.31465548, 0.383573, 0.7433543];
        MODy = [0.6602395, 0.5271141, 0.51731443];
        MODz = [0.68196464, 0.7583024, 0.4240403];
    elif StainingVectorID==18:
        # "AstraBlue Fuchsin"
        MODx = [0.92045766, 0.13336428, 0.0];
        MODy = [0.35425216, 0.8301452, 0.0];
        MODz = [0.16511545, 0.5413621, 0.0];
    elif StainingVectorID==19:
        # "RGB"
        #MODx = [0.0, 1.0, 1.0];
        #MODy = [1.0, 0.0, 1.0];
        #MODz = [1.0, 1.0, 0.0];
        MODx = [0.001, 1.0, 1.0];
        MODy = [1.0, 0.001, 1.0];
        MODz = [1.0, 1.0, 0.001];
    elif StainingVectorID==20:
        # "CMY"
        MODx = [1.0, 0.0, 0.0];
        MODy = [0.0, 1.0, 0.0];
        MODz = [0.0, 0.0, 1.0];
    elif StainingVectorID==21:
        # "User values"
        MODx = [SM(1,1), SM(1,2), SM(1,3)];
        MODy = [SM(2,1), SM(2,2), SM(2,3)];
        MODz = [SM(3,1), SM(3,2), SM(3,3)];

    # %% COEFFICIENTS NORMALIZATION:
        
    len  = [0, 0, 0];
    cosx = [0, 0, 0];
    cosy = [0, 0, 0];
    cosz = [0, 0, 0];
    for i in range(3):
        len[i] = math.sqrt(MODx[i]*MODx[i] + MODy[i]*MODy[i] + MODz[i]*MODz[i]); # Normalization to have the lenght of the column equal to 1.
        if (len[i] != 0):
            cosx[i] = MODx[i]/len[i];
            cosy[i] = MODy[i]/len[i];
            cosz[i] = MODz[i]/len[i];

    # translation matrix
    if (cosx[1]==0.0):
        if (cosy[1]==0.0):
            if (cosz[1]==0.0):
                #2nd colour is unspecified
                cosx[1]=cosz[0];
                cosy[1]=cosx[0];
                cosz[1]=cosy[0];
    
    if cosx[2]==0.0:
        if cosy[2]==0.0:
            if cosz[2]==0.0:
                #3rd colour is unspecified
                if doIcross==1:
                    cosx[2] = cosy[0] * cosz[1] - cosz[0] * cosy[1];
                    cosy[2] = cosz[0] * cosx[1] - cosx[0] * cosz[1];
                    cosz[2] = cosx[0] * cosy[1] - cosy[0] * cosx[1];
                else:
                    if ((cosx[0]*cosx[0] + cosx[1]*cosx[1])> 1):
                        #Colour_3 has a negative R component
                        cosx[2]=0.0;
                    else:
                        cosx[2]=math.sqrt(1.0-(cosx[0]*cosx[0])-(cosx[1]*cosx[1]));


                    if ((cosy[0]*cosy[0] + cosy[1]*cosy[1])> 1):
                        #Colour_3 has a negative G component
                        cosy[2]=0.0;
                    else:
                        cosy[2]=math.sqrt(1.0-(cosy[0]*cosy[0])-(cosy[1]*cosy[1]));
    
                    if ((cosz[0]*cosz[0] + cosz[1]*cosz[1])> 1):
                        #Colour_3 has a negative B component
                        cosz[2]=0.0;
                    else:
                        cosz[2]=math.sqrt(1.0-(cosz[0]*cosz[0])-(cosz[1]*cosz[1]));

    leng = math.sqrt(cosx[2]*cosx[2] + cosy[2]*cosy[2] + cosz[2]*cosz[2]);
    if (leng != 0 and leng != 1):
        cosx[2]= cosx[2]/leng;
        cosy[2]= cosy[2]/leng;
        cosz[2]= cosz[2]/leng;
    
    COS3x3Mat = np.array([
                         [cosx[0], cosy[0], cosz[0]],
                         [cosx[1], cosy[1], cosz[1]],
                         [cosx[2], cosy[2], cosz[2]]
                         ])

    # %% MATRIX Q USED FOR THE COLOUR DECONVOLUTION:
        
    # Check the determinant to understand if the matrix is invertible
    if np.linalg.det(COS3x3Mat) >= -0.001 and np.linalg.det(COS3x3Mat) <= 0.001:
        # Check column 1
        if (COS3x3Mat[0, 0] + COS3x3Mat[1, 0] + COS3x3Mat[2, 0]==0):
            cosx[0] = 0.001;
            cosx[1] = 0.001;
            cosx[2] = 0.001;
        # Check column 2
        if (COS3x3Mat[0, 1] + COS3x3Mat[1, 1] + COS3x3Mat[2 ,1]==0):
            cosy[0] = 0.001;
            cosy[1] = 0.001;
            cosy[2] = 0.001;
        # Check column 3
        if (COS3x3Mat[0, 2] + COS3x3Mat[1, 2] + COS3x3Mat[2, 2]==0):
            cosz[0] = 0.001;
            cosz[1] = 0.001;
            cosz[2] = 0.001;
        # Check row 1
        if (COS3x3Mat[0, 0] + COS3x3Mat[0, 1] + COS3x3Mat[0, 2]==0):
            cosx[0] = 0.001;
            cosy[0] = 0.001;
            cosz[0] = 0.001;
        # Check row 2
        if (COS3x3Mat[1, 0] + COS3x3Mat[1, 1] + COS3x3Mat[1, 2]==0):
            cosx[1] = 0.001;
            cosy[1] = 0.001;
            cosz[1] = 0.001;
        # Check row 3
        if (COS3x3Mat[2, 0] + COS3x3Mat[2, 1] + COS3x3Mat[2, 2]==0):
            cosx[2] = 0.001;
            cosy[2] = 0.001;
            cosz[2] = 0.001;
        # Check diagonal 1
        if (COS3x3Mat[0, 0] + COS3x3Mat[1, 1] + COS3x3Mat[2, 2]==0):
            cosx[0] = 0.001;
            cosy[1] = 0.001;
            cosz[2] = 0.001;
        # Check diagonal 2
        if (COS3x3Mat[0, 2] + COS3x3Mat[1, 1] + COS3x3Mat[2, 0]==0):
            cosz[0] = 0.001;
            cosy[1] = 0.001;
            cosx[2] = 0.001;
    
        COS3x3Mat = np.array([
                    [cosx[0], cosy[0], cosz[0]],
                    [cosx[1], cosy[1], cosz[1]],
                    [cosx[2], cosy[2], cosz[2]]
                    ])   
    
        if np.linalg.det(COS3x3Mat) >= -0.001 and np.linalg.det(COS3x3Mat) <= 0.001:
            for k in range(3):
                if (cosx[k]==0): cosx[k]=0.001;
                if (cosy[k]==0): cosy[k]=0.001;
                if (cosz[k]==0): cosy[k]=0.001;
        
            COS3x3Mat = np.array([
                    [cosx[0], cosy[0], cosz[0]],
                    [cosx[1], cosy[1], cosz[1]],
                    [cosx[2], cosy[2], cosz[2]]
                    ])   
                 
            if np.linalg.det(COS3x3Mat) >= -0.001 and np.linalg.det(COS3x3Mat) <= 0.001:         
                print('WARNING: the vector matrix is non invertible! So, the images of the stainings (e.r. images with names: Stain0#_transmittance, and Stain0#_LUT) are OK, but the images with name "Img#_back" are unreliable!');

    # Matrix inversion (I double check: it works!)
    # NOTE: this is the code to mathematically invert a 3x3 matrix without
    # calling the Matlab function "M3x3inv = inv(M3x3)".
    # NOTE: Q3x3Mat = inv(COS3x3Mat);
    # NOTE: COS3x3Mat = inv(Q3x3Mat);
    A = cosy[1] - cosx[1] * cosy[0] / cosx[0];
    V = cosz[1] - cosx[1] * cosz[0] / cosx[0];
    C = cosz[2] - cosy[2] * V/A + cosx[2] * (V/A * cosy[0] / cosx[0] - cosz[0] / cosx[0]);
    q2 = (-cosx[2] / cosx[0] - cosx[2] / A * cosx[1] / cosx[0] * cosy[0] / cosx[0] + cosy[2] / A * cosx[1] / cosx[0]) / C;
    q1 = -q2 * V / A - cosx[1] / (cosx[0] * A);
    q0 = 1.0 / cosx[0] - q1 * cosy[0] / cosx[0] - q2 * cosz[0] / cosx[0];
    q5 = (-cosy[2] / A + cosx[2] / A * cosy[0] / cosx[0]) / C;
    q4 = -q5 * V / A + 1.0 / A;
    q3 = -q4 * cosy[0] / cosx[0] - q5 * cosz[0] / cosx[0];
    q8 = 1.0 / C;
    q7 = -q8 * V / A;
    q6 = -q7 * cosy[0] / cosx[0] - q8 * cosz[0] / cosx[0];
    # Q3x3Mat = [q0, q3, q6; q1, q4, q7; q2, q5, q8]; # THIS SHOULD BE THE ONE IN THE JAVA CODE
    Q3x3Mat = np.array([
                    [q0, q3, q6],
                    [q1, q4, q7],
                    [q2, q5, q8]
                    ])   
    Q3x3MatInverted = COS3x3Mat; # NOTE: "Q3x3MatInverted = inv(Q3x3Mat)"

    # %% TRANSMITTANCE COMPUTATION:
        
    for r in range(rowR):
        for c in range (colR):
            RGB1 = [ImgR[r,c], ImgG[r,c], ImgB[r,c]];
            if Const1ForNoLog0==0:
                for i in range(3):
                    if RGB1[i]==0:
                        RGB1[i]=1;
                
            # type(RGB1)
            # Version1
            for i in range(np.size(RGB1)): 
                RGB1[i] = (Const1ForNoLog0 + RGB1[i])/(255+Const1ForNoLog0)
            ACC = -np.log(RGB1);
            Dye01Dye02Dye03_Transmittance_v1 = 255*np.exp(-np.dot(ACC.T, Q3x3Mat));
        
            # Creation of the single mono-channels for the transmittance
            Dye01_transmittance[r,c] = Dye01Dye02Dye03_Transmittance_v1[0];
            Dye02_transmittance[r,c] = Dye01Dye02Dye03_Transmittance_v1[1];
            Dye03_transmittance[r,c] = Dye01Dye02Dye03_Transmittance_v1[2];
        
    # %% LUT COMPUTATION:

    rLUT = (np.zeros((256,3), dtype='float'));
    gLUT = (np.zeros((256,3), dtype='float'));
    bLUT = (np.zeros((256,3), dtype='float'));
    for i in range(3): # 1:3
        for j in range(256): # 0:255
            if cosx[i]<0:
                rLUT[255-j, i] = 255 + (j * cosx[i]);
            else:
                rLUT[255-j, i] = 255 - (j * cosx[i]);
    
            if cosy[i]<0:
                gLUT[255-j, i] = 255 + (j * cosy[i]);
            else:
                gLUT[255-j, i] = 255 - (j * cosy[i]);

        
            if cosz[i]<0:
                bLUT[255-j, i] = 255 + (j * cosz[i]);
            else:
                bLUT[255-j, i] = 255 - (j * cosz[i]);

    # Compute LUT in the format of ImageJ/Fiji
    LUTdye01 = np.zeros((256, 3))
    LUTdye02 = np.zeros((256, 3))
    LUTdye03 = np.zeros((256, 3))
    LUTdye01[:,0] = rLUT[:,0];
    LUTdye01[:,1] = gLUT[:,0];
    LUTdye01[:,2] = bLUT[:,0];
    LUTdye02[:,0] = rLUT[:,1];
    LUTdye02[:,1] = gLUT[:,1];
    LUTdye02[:,2] = bLUT[:,1];
    LUTdye03[:,0] = rLUT[:,2];
    LUTdye03[:,1] = gLUT[:,2];
    LUTdye03[:,2] = bLUT[:,2];
    LUTdye01 = LUTdye01/255;
    LUTdye02 = LUTdye02/255;
    LUTdye03 = LUTdye03/255;

    # %% REMOVE THE CONTRIBUTION OF A STAINING FROM THE RGB IMAGE:
    
    # Select the stain to be removed:
    if DyeToBeRemovedID == 1:
        Dye01_transmittance = (255*np.ones((rowR, colR, 1), dtype='double'));
    elif DyeToBeRemovedID == 2:
        Dye02_transmittance = (255*np.ones((rowR, colR, 1), dtype='double'));
    elif DyeToBeRemovedID == 3:
        Dye03_transmittance = (255*np.ones((rowR, colR, 1), dtype='double'));
  
    # Use the Q3x3MatInverted to go back in RGB
    for r in range(rowR): #= 1:rowR
        for c in range(colR): #= 1:colR
            Dye01Dye02Dye03_transmittance = [Dye01_transmittance[r,c], Dye02_transmittance[r,c], Dye03_transmittance[r,c]];
            for i in range(np.size(Dye01Dye02Dye03_transmittance)): 
                Dye01Dye02Dye03_transmittance[i] = (Const1ForNoLog0 + Dye01Dye02Dye03_transmittance[i])/(255+Const1ForNoLog0)
            ACC2 = np.dot(-np.log(Dye01Dye02Dye03_transmittance), Q3x3MatInverted);
        
            RGB_backNoNorm = np.exp(-ACC2);
            RGB_back = (255*RGB_backNoNorm);
            ImgR_back[r, c] = RGB_back[0];
            ImgG_back[r, c] = RGB_back[1];
            ImgB_back[r, c] = RGB_back[2];

    ImgR_back[ImgR_back>255] = 255;
    ImgG_back[ImgR_back>255] = 255;
    ImgB_back[ImgR_back>255] = 255;
    ImgR_back[ImgR_back<0] = 0;
    ImgG_back[ImgR_back<0] = 0;
    ImgB_back[ImgR_back<0] = 0;
    ImgR_back = np.floor(ImgR_back); # Use this to obtain the same values of the ImageJ Colour Deconvolution plugin. Basically, it is a uint8 conversion with rounded values.
    ImgG_back = np.floor(ImgG_back); # Use this to obtain the same values of the ImageJ Colour Deconvolution plugin. Basically, it is a uint8 conversion with rounded values.
    ImgB_back = np.floor(ImgB_back); # Use this to obtain the same values of the ImageJ Colour Deconvolution plugin. Basically, it is a uint8 conversion with rounded values.

    #dist = cv2.normalize(Dye01_transmittance, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    dist1 = cv2.convertScaleAbs(Dye01_transmittance)
    #cv2.imshow('window', dist1); cv2.waitKey(0); cv2.destroyAllWindows()
    return dist1

##########################################
import glob
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
###############################
def split_patients(path):
    patients = []
    classes = []
    # Read Patients and Classes
    for filename in tqdm(glob.iglob(path + '/**/*.png' , recursive=True) , desc = "Reading"):
        class_type = filename.split('/')[6]
        patient = filename.split('/')[7]
        if patient not in patients:
            patients.append(patient)
            classes.append(class_type)
    
    # Be sure that splitting is fair
    train_patient = []
    test_patient = []
    train_classes = []
    test_classes = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)

    # Split
    for train_index, test_index in sss.split(patients, classes):
        for idx in train_index.tolist():
            train_patient.append(patients[idx])
            train_classes.append(classes[idx])
        for idx in test_index.tolist():
            test_patient.append(patients[idx])
            test_classes.append(classes[idx])

    print("train_classes: " , set(train_classes))
    print("test_classes: " , set(test_classes))
    return train_patient , test_patient 

###############################

def read_data(path):
    class_dict = {
        'benign':0,
        'malignant' : 1
    }
    train_dataset = []
    test_dataset = []
    for filename in tqdm(glob.iglob(path + '/**/*.png' , recursive=True) , desc = "Reading"):
        image_class = filename.split('/')[4] 
        patient = filename.split('/')[7] 
        resolution =   filename.split('/')[8] 
        if resolution == '40X':
            image = cv2.imread(filename)
            if patient in train_patient:
                train_dataset.append((image , class_dict[image_class]))
            elif patient in test_patient:
                test_dataset.append((image , class_dict[image_class]))
    return train_dataset , test_dataset  


#####################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import imageio

I_t_RGB = cv2.imread('BreaKHis_only40X/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-017.png')

def My_stain_normalization(I_s_RGB):
    M1 = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])
    m1 = np.array([[1/np.sqrt(3), 0, 0],[0, 1/np.sqrt(6), 0],[0, 0, 1/np.sqrt(2)]])
    m2 = np.array([[1, 1, 1],[1, 1, -2],[1, -1, 0]])
    M2 = m1.dot(m2)

    I_s_lms = np.zeros(np.shape(I_s_RGB))
    I_t_lms = np.zeros(np.shape(I_s_RGB))
    I_s_LMS = np.zeros(np.shape(I_s_RGB))
    I_t_LMS = np.zeros(np.shape(I_s_RGB))
    LAB_s = np.zeros(np.shape(I_s_RGB))
    LAB_t = np.zeros(np.shape(I_s_RGB))
    LAB_o = np.zeros(np.shape(I_s_RGB))
    I_o_LMS = np.zeros(np.shape(I_s_RGB))
    I_o_RGB = np.zeros(np.shape(I_s_RGB))

    for i in range(np.size(I_s_RGB,0)):
        for j in range(np.size(I_s_RGB,1)):
            ds = M1.dot(I_s_RGB[i,j,:])
            ls = np.log10(ds + np.finfo(float).eps)
            I_s_lms[i,j,:] = ds
            I_s_LMS[i,j,:] = ls
            
            dt = M1.dot(I_t_RGB[i,j,:])  
            lt = np.log10(dt + np.finfo(float).eps)
            I_t_lms[i,j,:] = dt        
            I_t_LMS[i,j,:] = lt
            
            
            Ds = M2.dot(I_s_LMS[i,j,:])
            LAB_s[i,j,:] = Ds
            
            Dt = M2.dot(I_t_LMS[i,j,:])
            LAB_t[i,j,:] = Dt

        
    M_s = np.array( [np.mean(LAB_s[:,:,0]), np.mean(LAB_s[:,:,1]), np.mean(LAB_s[:,:,2])] )
    M_t = np.array( [np.mean(LAB_t[:,:,0]), np.mean(LAB_t[:,:,1]), np.mean(LAB_t[:,:,2])] )

    V_s = np.array( [np.std(LAB_s[:,:,0]), np.std(LAB_s[:,:,1]), np.std(LAB_s[:,:,2])] )   
    V_t = np.array( [np.std(LAB_t[:,:,0]), np.std(LAB_t[:,:,1]), np.std(LAB_t[:,:,2])])


    m1 = np.array([[1, 1, 1],[1, 1, -1],[1, -2, 0]])
    m2 = np.array([[np.sqrt(3)/3, 0, 0],[0, np.sqrt(6)/6, 0],[0, 0, np.sqrt(2)/2]])
    M3 = m1.dot(m2)
    M4 = np.array([[4.4679, -3.5873, 0.1193],[-1.2186, 2.3809, -0.1624],[0.0497, -0.2439, 1.2045]])

    for i in range(np.size(I_s_RGB,0)):
        for j in range(np.size(I_s_RGB,1)):
            lab_p = LAB_s[i,j,:] - M_s
            lo1 =  ( V_t/(V_s+np.finfo(float).eps) )*lab_p + M_t 
            LAB_o[i,j,:] = lo1
            
            Lo = M3.dot(LAB_o[i,j,:])
            I_o_LMS[i,j,:] = Lo
        
            lo = np.power(10, Lo) 
            do = M4.dot(lo)
            I_o_RGB[i,j,:] = do
    return I_o_RGB


def Normalize(dataset , storage_path):
    counter = 0
    for image , label in tqdm(dataset):
        normalized_image = My_stain_normalization(image)
        normalized_image = normalized_image/normalized_image.max()
        normalized_image = 255 * normalized_image # Now scale by 255
        normalized_image = normalized_image.astype(np.uint8)
        cv2.imwrite(storage_path +'/' +str(label) + '_' + str(counter) + '.jpg', normalized_image)
        counter +=1 
        #new_dataset.append((normalized_image , label))

#############################################
def read_normalized(path):
    dataset = []
    for filename in tqdm(glob.iglob(path + '/**/*.jpg' , recursive=True) , desc = "Reading"):
        image_name = filename.split('/')[3] 
        label = image_name.split('_')[0]
        image = cv2.imread(filename)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (100,100))
        #image = cv2.medianBlur(image, 5)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
        #image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        image =  transforms.ToPILImage()(image)
        image = transforms.Resize((299 , 299))(image)
        image = transforms.ToTensor()(image)
        dataset.append((image , int(label)))
    return dataset


######################################
def preproces(dataset):
    new_dataset = []
    for image , label in dataset:
        image = transforms.ToPILImage()(image)
        #image = transforms.Resize((64 , 64))(image)
        image = transforms.ToTensor()(image)
        new_dataset.append((image, label))
    return new_dataset


########################################
def get_counts(dataset):
    counts = [0] * 2
    for element in dataset:
        t = element[1]
        if   t == 0: counts[0] += 1
        elif t == 1: counts[1] += 1
    return counts
##########################################
def transformation0(dataset):
    i = 0
    j = 0
    for image , img_name, label in dataset: 
        if label == 0:
            j += 1
        if label == 1:
            i += 1
    new_dataset = []
    k = 0
    while k == 0:
          for image , img_name, label in dataset: 
              if label == 0:
                 if j<i:
                    image = transforms.ToPILImage()(image)
                    image = transforms.RandomHorizontalFlip()(image)
                    image = transforms.RandomRotation(15)(image)
                    image = transforms.ToTensor()(image)
                    image_name = "a" + img_name
                    new_dataset.append((image, image_name, label))
                    j += 1

              if label == 1:
                 if i<j:
                    image = transforms.ToPILImage()(image)
                    image = transforms.RandomHorizontalFlip()(image)
                    image = transforms.RandomRotation(15)(image)
                    image = transforms.ToTensor()(image)
                    image_name = "a" + img_name
                    new_dataset.append((image, image_name, label))
                    i += 1
          if i == j:
             k = 1
        
    return new_dataset

##########################################
def transformation_all(dataset):
    new_dataset = []
    for image , label in dataset: 
        image = transforms.ToPILImage()(image)
        image = transforms.RandomHorizontalFlip()(image)
        image = transforms.RandomAffine(degrees = 90, translate= (0.2 , 0.2))(image)
        image = transforms.ToTensor()(image)
        new_dataset.append((image , label))
    return new_dataset
