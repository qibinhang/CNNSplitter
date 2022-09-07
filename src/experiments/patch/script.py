import os

CUDA = 'CUDA_VISIBLE_DEVICES=MIG-GPU-86993941-1e8e-3893-ccab-cc6bf0ea5a8c/11/0'


def run(target_epochs, model, dataset, exp_type):
    for i, target_epoch in enumerate(target_epochs):
        print('=' * 40 + f' PATCH {i} ' + '=' * 40)
        cmd = f'{CUDA} python apply_patch.py --model {model} --dataset {dataset} --exp_type {exp_type} --target_epoch {target_epoch} --target_class {i}'
        os.system(cmd)
        print('\n\n')


# 1. CIFAR-10
# 1.1 Weak Model
# print('='*20 + 'Weak SimCNN  |  CIFAR-10' + '='*20)
# best_epoch = [99, 99, 98, 99, 85, 96, 65, 94, 92, 96]  # simcnn
# run(best_epoch, 'simcnn', 'cifar10', 'weak')

print('='*20 + 'Weak ResCNN  |  CIFAR-10' + '='*20)
best_epoch = [87, 90, 74, 98, 76, 97, 89, 98, 78, 91]  # rescnn
run(best_epoch, 'rescnn', 'cifar10', 'weak')


# 1.2 overfitting
# print('='*20 + 'Overfitting SimCNN  |  CIFAR-10' + '='*20)
# best_epoch = [169, 177, 123, 118, 137, 137, 138, 140, 135, 169]  # simcnn
# run(best_epoch, 'simcnn', 'cifar10', 'poor_fit')

print('='*20 + 'Overfitting ResCNN  |  CIFAR-10' + '='*20)
best_epoch = [131, 105, 120, 126, 136, 127, 144, 124, 136, 109]  # rescnn
run(best_epoch, 'rescnn', 'cifar10', 'poor_fit')


# 1.3 Underfitting
# print('='*20 + 'Underfitting SimCNN  |  CIFAR-10' + '='*20)
# best_epoch = [84, 88, 61, 59, 68, 68, 69, 70, 67, 84]  # simcnn
# run(best_epoch, 'simcnn', 'cifar10', 'poor_fit')

print('='*20 + 'Underfitting ResCNN  |  CIFAR-10' + '='*20)
best_epoch = [65, 52, 60, 63, 67, 63, 72, 62, 68, 54]  # rescnn
run(best_epoch, 'rescnn', 'cifar10', 'poor_fit')


# 3. SVHN_5
# 3.1 Weak Model
print('='*20 + 'Weak SimCNN  |  SVHN_5' + '='*20)
best_epoch = [77, 77, 85, 97, 89]  # simcnn
run(best_epoch, 'simcnn', 'svhn_5', 'weak')

print('='*20 + 'Weak ResCNN  |  SVHN_5' + '='*20)
best_epoch = [88, 80, 84, 91, 68]  # rescnn
run(best_epoch, 'rescnn', 'svhn_5', 'weak')


# 3.2 overfitting
print('='*20 + 'Overfitting SimCNN  |  SVHN_5' + '='*20)
best_epoch = [53, 31, 59, 62, 48]  # simcnn
run(best_epoch, 'simcnn', 'svhn_5', 'poor_fit')

print('='*20 + 'Overfitting ResCNN  |  SVHN_5' + '='*20)
best_epoch = [59, 59, 49, 57, 61]  # rescnn
run(best_epoch, 'rescnn', 'svhn_5', 'poor_fit')


# 3.3 Underfitting
print('='*20 + 'Underfitting SimCNN  |  SVHN_5' + '='*20)
best_epoch = [26, 15, 29, 31, 24]  # simcnn
run(best_epoch, 'simcnn', 'svhn_5', 'poor_fit')

print('='*20 + 'Underfitting ResCNN  |  SVHN_5' + '='*20)
best_epoch = [29, 29, 24, 28, 30]  # rescnn
run(best_epoch, 'rescnn', 'svhn_5', 'poor_fit')
