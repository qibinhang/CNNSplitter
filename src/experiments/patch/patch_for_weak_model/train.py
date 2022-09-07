import argparse
import torch
import sys
import copy
import torchvision.transforms as transforms
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from simcnn_weak import SimCNNWeak
from rescnn_weak import ResCNNWeak
from simcnn_svhn_weak import SimCNNSVHNWeak
from rescnn_svhn_weak import ResCNNSVHNWeak
from tqdm import tqdm
sys.path.append('../../..')
from utils.configure_loader import load_configure
from utils.checker import check_dir
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
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        normalize])
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
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        normalize])

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


def train(model, train_loader, val_loader, model_save_dir):
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    best_acc, best_epoch = 0.0, 0
    best_model = None
    optimization = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(n_epochs):
        print(f'epoch {epoch}')
        print('-'*80)

        # train
        epoch_train_loss = []
        epoch_train_acc = []
        model.train()
        for batch_inputs, batch_labels in tqdm(train_loader, ncols=100, desc='train'):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            optimization.zero_grad()
            loss = loss_func(outputs, batch_labels)
            loss.backward()
            optimization.step()
            epoch_train_loss.append(loss.detach())

            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == batch_labels)
            epoch_train_acc.append(torch.div(acc, batch_labels.shape[0]))
        print(f"train_loss: {sum(epoch_train_loss)/len(epoch_train_loss):.2f}")
        print(f"train_acc : {sum(epoch_train_acc)/len(epoch_train_acc) * 100:.2f}%")

        # val
        epoch_val_acc = []
        model.eval()
        with torch.no_grad():
            for batch_inputs, batch_labels in tqdm(val_loader, ncols=100, desc='valid'):
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                outputs = model(batch_inputs)
                pred = torch.argmax(outputs, dim=1)
                acc = torch.sum(pred == batch_labels)
                epoch_val_acc.append(torch.div(acc, batch_labels.shape[0]))
            val_acc = sum(epoch_val_acc)/len(epoch_val_acc)
        print(f"val_acc   : {val_acc * 100:.2f}%")
        print()

        torch.save(model.state_dict(), f'{model_save_dir}/epoch_{epoch}.pth')
        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())

    print(f"best_epoch: {best_epoch}")
    print(f"best_acc  : {best_acc * 100:.2f}%")
    model.load_state_dict(best_model)
    return model


def test(model, test_loader):
    epoch_acc = []
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == batch_labels)
            epoch_acc.append(torch.div(acc, batch_labels.shape[0]))
    print(f"\nTest Accuracy: {sum(epoch_acc) / len(epoch_acc) * 100:.2f}%")


def main():
    print(f'Using {device}')
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

    for target_class in range(num_classes):
        print('='*40 + f'Target Class {target_class}' + '='*40)
        if dataset_name == 'cifar10':
            if model_name == 'simcnn':
                model = SimCNNWeak(num_classes=num_classes).to(device)
            else:
                model = ResCNNWeak(num_classes=num_classes).to(device)
        else:
            if model_name == 'simcnn':
                model = SimCNNSVHNWeak(num_classes=num_classes).to(device)
            else:
                model = ResCNNSVHNWeak(num_classes=num_classes).to(device)
        print(model)

        model_save_dir = f'{configs.data_dir}/patch/weak/{model_name}_{dataset_name}/target_class_{target_class}'
        check_dir(model_save_dir)

        train_loader, val_loader = load_dataset(data_dir_1, data_dir_2, is_train=True, mix_labels=[target_class],
                                                batch_size=batch_size, num_workers=0, pin_memory=False)
        test_loader = load_dataset(data_dir_1, data_dir_2, is_train=False, mix_labels=[target_class],
                                   batch_size=batch_size, num_workers=1, pin_memory=True)

        model = train(model, train_loader, val_loader, model_save_dir)
        model.eval()
        test(model, test_loader)
        print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn_5'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--evaluation', action='store_true')
    args = parser.parse_args()
    print(args)
    print()

    model_name = args.model
    dataset_name = args.dataset
    lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.epochs
    evaluation = args.evaluation
    main()
