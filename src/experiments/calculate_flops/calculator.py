import argparse
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
import sys
sys.path.append('../..')
from utils.configure_loader import load_configure
from utils.module_tools import load_modules, load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_flop(model, img_size, show_details: bool = False):
    x = torch.randn(1, 3, img_size, img_size)
    fca = FlopCountAnalysis(model, x)
    flops = fca.total()
    if show_details:
        print(flop_count_table(fca))
    return flops


def main():
    # FLOPs of the strong model
    configs = load_configure(model_name, dataset_name=dataset_name if dataset_name != 'svhn_5' else 'svhn')
    model = load_model(model_name, num_classes=10)

    trained_model_path = f"{configs.trained_model_dir}/{configs.trained_entire_model_name}"
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    model_flops = calculate_flop(model, img_size=32)

    # FLOPs of the module (i.e., patch) from the strong model
    configs = load_configure(model_name, dataset_name)
    modules = load_modules(configs)
    target_module = modules[target_class][0]
    module_flops = calculate_flop(target_module, img_size=32)

    print(f'Model : {model_flops/1e6:.2f}M')
    print(f'Module: {module_flops/1e6:.2f}M')
    print(f'Overhead Module/Model: {module_flops/model_flops:.2%}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn_5'])
    parser.add_argument('--target_class', choices=list(range(10)), type=int)
    args = parser.parse_args()
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    target_class = args.target_class

    main()
