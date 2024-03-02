
############## Tune these
num_classes = 8
n_fold = '1'

base_architecture = 'vgg19'
experiment_run = '03'

img_size = 300  # 224
prototype_shape = (16, 128, 1, 1) # 16 = num_classes*num_prototype per each class

prototype_activation_function = 'log'
add_on_layers_type = 'regular' # two additional 1*1 convolutional layers


magnification = '40x'
train_test = '/test'
fold = 'fold' + n_fold + '_' + magnification
#data_path = './datasets_exp_' + str(num_classes) + '/' + fold + '/'
data_path = '/content/drive/My Drive/pseudo-class generation/datasets_exp_' + str(num_classes) + '/' + fold + '/' 
train_dir = data_path + 'train_push_balanced/train_augmented/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train_push/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 80

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5 # We reduced the learning rate by a factor of 0:1 every 5 epochs

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08, #-0.08
    'l1': 1e-4,  # l1 is the loss comonent of last F.C. layer
}

num_train_epochs = 31
num_warm_epochs = 5

push_start = 10 #we performed prototype projection and convex optimization of the last layer (for 20 iterations) every two times we reduced the learning rate
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
