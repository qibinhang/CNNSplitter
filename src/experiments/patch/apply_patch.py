import argparse
import copy
import sys
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from patch_for_weak_model.simcnn_weak import SimCNNWeak
from patch_for_weak_model.rescnn_weak import ResCNNWeak
from patch_for_weak_model.simcnn_svhn_weak import SimCNNSVHNWeak
from patch_for_weak_model.rescnn_svhn_weak import ResCNNSVHNWeak
from patch_for_poor_fit.simcnn_poor_fit import SimCNNPoorFit
from patch_for_poor_fit.rescnn_poor_fit import ResCNNPoorFit
sys.path.append('../..')
from utils.configure_loader import load_configure
from utils.module_tools import load_modules, load_range_of_module_output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cifar10_mixed(data_dir_1, data_dir_2, is_train, mix_labels, batch_size=64, num_workers=0, pin_memory=False):
    """
    select {0: apple, 2: baby, 5: bed, 8: bicycle, 9: bottle, 12: bridge, 15: camel, 22: clock, 70: rose} from CIFAR-100
    """
    cifar100_labels = [0, 2, 5, 8, 9, 12, 15, 22, 70]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    normalize])
    if is_train:
        # CIFAR-100
        train_cifar100 = torchvision.datasets.CIFAR100(root=data_dir_1, train=True, transform=transform)

        # sample the data belonging to the 9 target classes.
        train_cifar100_targets = np.array(train_cifar100.targets)
        idx = np.isin(train_cifar100_targets, cifar100_labels)
        target_label = train_cifar100_targets[idx]
        trans_label = [cifar100_labels.index(i) for i in target_label]
        train_cifar100.targets = trans_label
        train_cifar100.data = train_cifar100.data[idx]

        idx = list(range(len(train_cifar100)))
        np.random.seed(1009)
        np.random.shuffle(idx)
        train_idx = idx[: int(0.8 * len(idx))]
        valid_idx = idx[int(0.8 * len(idx)):]

        train_cifar100_set = copy.deepcopy(train_cifar100)
        train_cifar100_set.targets = [train_cifar100.targets[idx] for idx in train_idx]
        train_cifar100_set.data = train_cifar100.data[train_idx]

        valid_cifar100_set = copy.deepcopy(train_cifar100)
        valid_cifar100_set.targets = [train_cifar100.targets[idx] for idx in valid_idx]
        valid_cifar100_set.data = train_cifar100.data[valid_idx]

        # CIFAR-10
        train_cifar10 = torchvision.datasets.CIFAR10(root=data_dir_2, train=True, transform=transform)
        train_cifar10_targets = np.array(train_cifar10.targets)
        idx = np.isin(train_cifar10_targets, mix_labels)
        target_label = train_cifar10_targets[idx]
        trans_label = [9 for _ in target_label]
        train_cifar10.targets = trans_label
        train_cifar10.data = train_cifar10.data[idx]

        idx = list(range(len(train_cifar10)))
        np.random.seed(1009)
        np.random.shuffle(idx)
        idx = idx[:500]  # each class has 500 training data in CIFAR100
        train_idx = idx[: int(0.8 * len(idx))]
        valid_idx = idx[int(0.8 * len(idx)):]

        train_cifar10_set = copy.deepcopy(train_cifar10)
        train_cifar10_set.targets = [train_cifar10.targets[idx] for idx in train_idx]
        train_cifar10_set.data = train_cifar10.data[train_idx]

        valid_cifar10_set = copy.deepcopy(train_cifar10)
        valid_cifar10_set.targets = [train_cifar10.targets[idx] for idx in valid_idx]
        valid_cifar10_set.data = train_cifar10.data[valid_idx]

        # mix
        train_set = train_cifar100_set
        train_set.targets = train_set.targets + train_cifar10_set.targets
        train_set.data = np.concatenate((train_set.data, train_cifar10_set.data), axis=0)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)

        valid_set = valid_cifar100_set
        valid_set.targets = valid_set.targets + valid_cifar10_set.targets
        valid_set.data = np.concatenate((valid_set.data, valid_cifar10_set.data), axis=0)
        valid_loader = DataLoader(valid_set, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, valid_loader
    else:
        # CIFAR-100
        test_cifar100 = torchvision.datasets.CIFAR100(root=data_dir_1, train=False, transform=transform)

        # sample the data belonging to the 9 target classes.
        test_cifar100_targets = np.array(test_cifar100.targets)
        idx = np.isin(test_cifar100_targets, cifar100_labels)
        target_label = test_cifar100_targets[idx]
        trans_label = [cifar100_labels.index(i) for i in target_label]
        test_cifar100.targets = trans_label
        test_cifar100.data = test_cifar100.data[idx]

        # CIFAR-10
        test_cifar10 = torchvision.datasets.CIFAR10(root=data_dir_2, train=False, transform=transform)
        test_cifar10_targets = np.array(test_cifar10.targets)
        idx = np.isin(test_cifar10_targets, mix_labels)
        target_label = test_cifar10_targets[idx]
        trans_label = [9 for _ in target_label]
        test_cifar10.targets = trans_label
        test_cifar10.data = test_cifar10.data[idx]
        test_cifar10.targets = test_cifar10.targets[:100]  # each class has 100 test data in CIFAR100
        test_cifar10.data = test_cifar10.data[:100]

        # mix
        test_set = test_cifar100
        test_set.targets = test_set.targets + test_cifar10.targets
        test_set.data = np.concatenate((test_set.data, test_cifar10.data), axis=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return test_loader


def load_svhn_mixed_self(data_dir_1, data_dir_2, is_train, mix_labels, batch_size=64, num_workers=0, pin_memory=False):
    """
    SVHN for weak: 6-9
    """
    svhn_weak_labels = list(range(6, 10))
    svhn_labels = svhn_weak_labels + mix_labels
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    normalize])
    if is_train:
        train_svhn = torchvision.datasets.SVHN(root=data_dir_2, split='train', transform=transform)
        # sample the data belonging to svhn_weak and mix_labels
        train_svhn_targets = train_svhn.labels
        idx = np.isin(train_svhn_targets, svhn_labels)
        target_label = train_svhn_targets[idx].tolist()
        trans_label = [svhn_labels.index(i) for i in target_label]
        train_svhn.labels = np.array(trans_label)
        train_svhn.data = train_svhn.data[idx]

        idx = list(range(len(train_svhn)))
        np.random.seed(1009)
        np.random.shuffle(idx)
        train_idx = idx[: int(0.8 * len(idx))]
        valid_idx = idx[int(0.8 * len(idx)):]

        train_svhn_set = copy.deepcopy(train_svhn)
        train_svhn_set.labels = train_svhn.labels[train_idx]
        train_svhn_set.data = train_svhn.data[train_idx]
        train_loader = DataLoader(train_svhn_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)

        valid_svhn_set = copy.deepcopy(train_svhn)
        valid_svhn_set.labels = train_svhn.labels[valid_idx]
        valid_svhn_set.data = train_svhn.data[valid_idx]
        valid_loader = DataLoader(valid_svhn_set, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, valid_loader
    else:
        test_svhn = torchvision.datasets.SVHN(root=data_dir_2, split='test', transform=transform)
        test_svhn_targets = test_svhn.labels
        idx = np.isin(test_svhn_targets, svhn_labels)
        target_label = test_svhn_targets[idx].tolist()
        trans_label = [svhn_labels.index(i) for i in target_label]
        test_svhn.labels = np.array(trans_label)
        test_svhn.data = test_svhn.data[idx]
        test_loader = DataLoader(test_svhn, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=pin_memory)
        return test_loader


@torch.no_grad()
def evaluate_model(model, test_dataset, print_confusion_matrix=False):
    predicts = []
    labels = []
    for batch_inputs, batch_labels in test_dataset:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        outputs = model(batch_inputs)
        pred = torch.argmax(outputs, dim=1)
        predicts.append(pred)
        labels.append(batch_labels)
    predicts = torch.cat(predicts)
    labels = torch.cat(labels)

    if print_confusion_matrix:
        get_confusion_matrix(predicts.cpu().numpy(), labels.cpu().numpy(), torch.max(labels).cpu().item())  # target class is always appended in the tail.


def get_confusion_matrix(predicts, labels, target):
    predicts_trans = predicts == target
    labels_trans = labels == target
    cm = confusion_matrix(labels_trans, predicts_trans)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp/(tp+fn)
    f_score = 2 * precision * recall / (precision + recall)
    print(f'precision: {precision * 100:.2f}%')
    print(f'recall: {recall * 100:.2f}%')
    print(f'f-score: {f_score:.4f}')


@torch.no_grad()
def apply_patch(target_class: int, model, modules, module_output_ranges, test_loader, print_confusion_matrix=False):
    total_predicts, total_labels = [], []
    for batch_inputs, batch_labels in test_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        model_outputs = model(batch_inputs)
        model_outputs = torch.softmax(model_outputs, dim=1)

        target_module = modules[target_class][0]
        module_output = target_module(batch_inputs)
        target_modules_output = module_output[:, target_class]
        target_modules_output = torch.unsqueeze(target_modules_output, dim=1)

        # min-max
        target_modules_output_range = module_output_ranges[target_class]
        target_modules_output = (target_modules_output - target_modules_output_range[0]) / (target_modules_output_range[1] - target_modules_output_range[0])

        final_outputs = torch.cat((model_outputs[:, :-1], target_modules_output), dim=1)  # always class 9 is from CIFAR 10
        final_pred = torch.argmax(final_outputs, dim=1)
        total_predicts.append(final_pred)
        total_labels.append(batch_labels)

    total_predicts = torch.cat(total_predicts)
    total_labels = torch.cat(total_labels)

    if print_confusion_matrix:
        get_confusion_matrix(total_predicts.cpu().numpy(), total_labels.cpu().numpy(), torch.max(total_labels).cpu().item())  # target class is always appended in the tail.


def main():
    configs = load_configure(model_name, dataset_name)
    cifar10_data_dir = f'{configs.data_dir}/dataset'
    cifar100_data_dir = f'{configs.data_dir}/dataset'
    svhn_data_dir = f'{configs.data_dir}/dataset/svhn'
    if dataset_name == 'cifar10':
        load_dataset = load_cifar10_mixed
        data_dir_1 = cifar100_data_dir
        data_dir_2 = cifar10_data_dir
        num_classes = 10
    else:
        load_dataset = load_svhn_mixed_self
        data_dir_1 = svhn_data_dir
        data_dir_2 = svhn_data_dir
        num_classes = 5

    if exp_type == 'weak':
        save_path = f'{configs.data_dir}/patch/weak/{model_name}_{dataset_name}/target_class_{target_class}/epoch_{target_epoch}.pth'
        if dataset_name == 'svhn_5':
            model = SimCNNSVHNWeak(num_classes=num_classes) if model_name == 'simcnn' else ResCNNSVHNWeak(
                num_classes=num_classes)
        else:
            model = SimCNNWeak(num_classes=num_classes) if model_name == 'simcnn' else ResCNNWeak(
                num_classes=num_classes)
    else:
        assert target_epoch >= 0
        save_path = f'{configs.data_dir}/patch/poor/{model_name}_{dataset_name}/target_class_{target_class}/epoch_{target_epoch}.pth'
        model = SimCNNPoorFit(num_classes=num_classes) if model_name == 'simcnn' else ResCNNPoorFit(
            num_classes=num_classes)
    print(f'model path: {save_path}')

    model = model.to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_loader = load_dataset(data_dir_1, data_dir_2, is_train=False, mix_labels=[target_class], batch_size=128)

    # before patch
    print('\n\nBefore Patching')
    evaluate_model(model, test_loader, print_confusion_matrix=True)

    # after patch
    print('\n\nAfter Patching')
    modules = load_modules(configs)
    module_output_ranges = load_range_of_module_output(configs.module_output_path, mode='min_max')
    apply_patch(target_class, model, modules, module_output_ranges, test_loader, print_confusion_matrix=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn_5'])
    parser.add_argument('--exp_type', choices=['weak', 'poor_fit'])
    parser.add_argument('--target_class', choices=list(range(10)), type=int)
    parser.add_argument('--target_epoch', type=int)
    args = parser.parse_args()
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    exp_type = args.exp_type
    target_class = args.target_class
    target_epoch = args.target_epoch

    main()