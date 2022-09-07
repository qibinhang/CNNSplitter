import argparse
import os
import sys
sys.path.append('..')
from utils.checker import check_dir
cuda_visible = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn', 'svhn_5'])

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    if model_name == 'simcnn':
        target_layers = list(range(13))
    elif model_name == 'rescnn':
        target_layers = list(range(12))
    else:
        raise ValueError

    log_dir = f'../../data/{model_name}_{dataset_name}/layer_sensitivity'
    check_dir(log_dir)
    print(f'save log in {os.path.abspath(log_dir)}')

    if cuda_visible != 0:
        cmd_cuda_visible = f'CUDA_VISIBLE_DEVICES={cuda_visible} '
    else:
        cmd_cuda_visible = ''

    for i, tl in enumerate(target_layers):
        print(f'Target Layer: {tl}')
        cmd = f'{cmd_cuda_visible}python -u ../preprocess/layer_sensitivity_analyzer.py ' \
              f'--model {model_name} --dataset {dataset_name} --target_conv {tl}'
        cmd = f'{cmd} > {log_dir}/layer_{tl}.log'
        print(f'{i+1}/{len(target_layers)}: {cmd}\n')
        os.system(cmd)
