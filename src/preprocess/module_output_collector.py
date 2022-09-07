import argparse
import torch
import sys
import _pickle as pickle
from tqdm import tqdm
sys.path.append('..')
from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.module_tools import load_modules
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn', 'svhn_5'])
    args = parser.parse_args()
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    configs = load_configure(model_name, dataset_name)
    load_dataset = get_dataset_loader(dataset_name)

    print(f"=== {configs.dataset_name} ===")
    print(f'best_generation = {configs.best_generation}')
    print(f'best_sol_ensemble = {configs.best_sol_ensemble}')
    print(f'log_idx = {configs.log_idx}')
    print(f'best_acc = {configs.best_acc}')
    print(f'best_diff = {configs.best_diff}\n')

    modules = [m[0] for m in load_modules(configs)]
    all_module_outputs = []

    with torch.no_grad():
        for target_class, each_module in enumerate(tqdm(modules, desc='collecting', ncols=100)):
            outputs = []
            train_dataset, valid_dataset = load_dataset(configs.dataset_dir, labels=[target_class], is_train=True,
                                                        is_random=False, batch_size=128, num_workers=1, pin_memory=True)
            for inputs, labels in valid_dataset:
                pred = each_module(inputs.to(device))
                outputs.append(pred[:, target_class])
            outputs = torch.cat(outputs, dim=0)
            all_module_outputs.append(outputs.cpu().numpy())
    with open(configs.module_output_path, 'wb') as f:
        pickle.dump(all_module_outputs, f)
