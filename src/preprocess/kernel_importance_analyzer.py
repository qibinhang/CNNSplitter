import argparse
import torch
import sys
import _pickle as pickle
sys.path.append('..')
from utils.dataset_loader import get_dataset_loader
from utils.model_loader import load_model
from utils.configure_loader import load_configure
from utils.feature_map_L1 import get_avg_L1_of_feature_maps
from utils.checker import check_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_kernel_importance(model, load_dataset, mode, dataset_dir, save_dir, target_classes):
    if mode == 'L1':
        importance_path = f'{save_dir}/avg_L1.pkl'
        kernel_importance = get_avg_L1_of_feature_maps(model, load_dataset, dataset_dir, target_classes)
    else:
        raise ValueError

    check_dir(save_dir)
    with open(importance_path, 'wb') as f:
        pickle.dump(kernel_importance, f)


def main():
    configs = load_configure(model_name, dataset_name)
    trained_model_path = f"{configs.trained_model_dir}/{configs.trained_entire_model_name}"
    kernel_importance_dir = configs.kernel_importance_dir
    mode = configs.kernel_importance_analyzer_mode
    dataset_dir = configs.dataset_dir
    check_dir(kernel_importance_dir)

    if dataset_name == 'svhn_5':
        target_classes = list(range(5))
    else:
        target_classes = list(range(10))
    model = load_model(model_name=model_name, num_classes=len(target_classes)).to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.eval()
    load_dataset = get_dataset_loader(dataset_name=dataset_name)

    generate_kernel_importance(model, load_dataset, mode, dataset_dir,
                               kernel_importance_dir, target_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn', 'svhn_5'])
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    main()
