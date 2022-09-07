import os

for model in ['simcnn', 'rescnn']:
    for dataset in ['cifar10', 'svhn_5']:
        tc_list = range(10) if dataset == 'cifar10' else range(5)

        for tc in tc_list:
            cmd = f'python -u calculator.py --model {model} --dataset {dataset} --target_class {tc}'
            os.system(cmd)
            print()

