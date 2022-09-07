import argparse
import torch
import sys
import _pickle as pickle
from tqdm import tqdm
sys.path.append('..')
from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.module_tools import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn'])
    args = parser.parse_args()
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    configs = load_configure(model_name, dataset_name)
    load_dataset = get_dataset_loader(dataset_name)
    model = load_model(model_name).to(device)
    model.load_state_dict(
        torch.load(f'{configs.trained_model_dir}/{configs.trained_entire_model_name}', map_location=device)
    )
    model_outputs = []
    with torch.no_grad():
        for target_class in tqdm(list(range(10)), desc='collecting', ncols=100):
            outputs = []
            _, valid_dataset = load_dataset(configs.dataset_dir, labels=[target_class], is_train=True, is_random=False,
                                            batch_size=128, num_workers=1, pin_memory=True)
            for inputs, labels in valid_dataset:
                pred = model(inputs.to(device))
                outputs.append(pred[:, target_class])
            outputs = torch.cat(outputs, dim=0)
            model_outputs.append(outputs.cpu().numpy())
    with open(f'{configs.workspace}/model_outputs.pkl', 'wb') as f:
        pickle.dump(model_outputs, f)
