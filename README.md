
This code package implements the Pseudo-Class Part Prototype Network (PCPPN) from the paper "Pseudo-Class Part Prototype Networks for Interpretable Breast Cancer Classification"
(to puplish in Scientific Report Journal 2024), by Mohammad Amin Choukali* (Urmia University), Mehdi Chehel Amirani Urmia University),
Morteza Valizadeh (Urmia University), Ata Abbasi (Urmia University of medicalsciences), and Majid Komeili (Carleton University)

**Prerequisites:** PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor)

*Recommended hardware*: Tesla T4 GPU (Google Colaboratory)

**Instructions for preparing the data:**
1. Download the BreakHist dataset, BreakHist_v1 (https://www.kaggle.com/datasets/ambarish/breakhis)
2. Open "./Data Revision and Division into 5 folds/"
3. Put BreakHist_v1 in directory "./Data Revision and Division into 5 folds"
4. Run 'revised_5folds_40X_generation.py'
5. Open "./pseudo-class generation/"
6. Put resulting "revised_foldk_40x" folders in the directory "./pseudo-class generation/dataset/"
7. Run the Jupyter Notebook 'DataAugmentation_for_BaseTrain.ipynb'
8. Run the Jupyter Notebook 'train_base_model.ipynb'
9. Run the Jupyter Notebook 'Clustering_stage.ipynb'

**Instructions for training the model:**
1. Open "./PCPPN/"
2. In settings.py, provide the appropriate strings for parameters/ directories including:
num_classes, n_fold, data_path, train_dir, test_dir, train_push_dir:
- num_classes is the number of pseudo-classes, 2*K, where K denotes the number of clusters
- n_fold is the number of data division fold {1, 2, 3, 4, 5} which you choose to use  
- data_path is where the dataset resides
- train_dir is the directory containing the augmented training set
- test_dir is the directory containing the test set
- train_push_dir is the directory containing the original (unaugmented) training set
3. Run the Jupyter Notebook 'Augment_data_PCPPN.ipynb'
4. Run the Jupyter Notebook 'main_PCPPN.ipynb'

**Instructions for finding the nearest patches to each prototype:**
1. Open "./PCPPN/"
2. Run the Jupyter Notebook 'global_analysis.ipynb' and supply the following arguments:
- gpuid is the GPU device ID(s) you want to use (optional, default '0')
- modeldir is the directory containing the model you want to analyze
- model is the filename of the saved model you want to analyze

