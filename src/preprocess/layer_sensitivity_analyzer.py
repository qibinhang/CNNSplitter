import argparse
import torch
import sys
sys.path.append('../')
from utils.kernel_importance_loader import load_kernel_importance
from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.model_loader import load_model
from utils.module_tools import extract_module, evaluate_ensemble_modules
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_module(target_class, conv_idx, top_ratio, model, kernel_importance):
    """
        target_class: int
        target_layer: list, indicate which conv_layers will be considered. e.g. [0, 1, 12]
        top_ratio: list, indicate the percentage of top kernels to be reserved. e.g. [0.9, 0.8, 0.1]
    """
    ki_on_target_class = kernel_importance[target_class]
    ki = ki_on_target_class[conv_idx]
    sorted_ki = list(sorted(zip(range(len(ki)), ki), key=lambda x: x[1], reverse=True))
    important_kernel_indices = [ki[0] for ki in sorted_ki[:int(len(sorted_ki) * top_ratio)]]
    conv_info = {}
    for conv_name in model._modules:
        if 'conv_' not in conv_name:
            continue
        if conv_name == f'conv_{conv_idx}':
            conv_info[conv_name] = important_kernel_indices
        else:
            if model.__class__.__name__ == 'SimCNN':
                conv_info[conv_name] = list(range(model._modules[conv_name].out_channels))
            elif model.__class__.__name__ == 'ResCNN':
                conv_info[conv_name] = list(range(model._modules[conv_name]._modules['0'].out_channels))
            else:
                raise ValueError

    if model.__class__.__name__ == 'ResCNN':
        if target_conv in (1, 5, 9):
            conv_info[f'conv_{target_conv + 2}'] = conv_info[f'conv_{target_conv}']
        elif target_conv in (3, 7, 11):
            conv_info[f'conv_{target_conv - 2}'] = conv_info[f'conv_{target_conv}']

    module, _ = extract_module(conv_info, model)
    return module


def main():
    configs = load_configure(model_name, dataset_name)
    trained_model_path = f"{configs.trained_model_dir}/{configs.trained_entire_model_name}"
    dataset_dir = configs.dataset_dir

    load_dataset = get_dataset_loader(dataset_name=dataset_name)
    if dataset_name == 'svhn_5':
        target_classes = list(range(5))
    else:
        target_classes = list(range(10))
    model = load_model(model_name=model_name, num_classes=len(target_classes)).to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    _, valid_dataset = load_dataset(dataset_dir, is_train=True, labels=target_classes, is_random=False,
                                    num_workers=2, pin_memory=True, batch_size=256)
    kernel_importance = load_kernel_importance(configs.kernel_importance_analyzer_mode,
                                               configs.kernel_importance_dir)
    print(f'Layer {target_conv}')
    print('='*50)
    for top_ratio in [i/10 for i in range(1, 11)]:
        ensemble_modules = []
        print(f'top_ratio: {top_ratio}')
        for tc in target_classes:
            module = init_module(tc, target_conv, top_ratio, model, kernel_importance)
            ensemble_modules.append(module.to(device))
        acc = evaluate_ensemble_modules(ensemble_modules, valid_dataset)
        print(f'acc: {acc}\n')
    print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn', 'svhn_5'])
    parser.add_argument('--target_conv', type=int)

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    target_conv = args.target_conv
    main()
