import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn'])

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset

    if model_name == 'simcnn':
        target_layers = list(range(13))
    elif model_name == 'rescnn':
        target_layers = list(range(12))
    else:
        raise ValueError

    format_results = [[] for _ in range(len(target_layers))]
    for tl in target_layers:
        path = f'../../data/{model_name}_{dataset_name}/layer_sensitivity/layer_{tl}.log'
        with open(path, 'r') as f:
            for line in f.readlines():
                if line.startswith('top_ratio:'):
                    ratio = float(line.strip().split()[1])
                elif line.startswith('acc:'):
                    acc = float(line.strip().split()[1])
                    format_results[tl].append(f'{ratio:.1f} {acc * 100:.2f}%')

    for layer_idx, each_layer_result in enumerate(format_results):
        result = '\n'.join(each_layer_result)
        print(f'Layer {layer_idx}')
        print(result)
        print()
