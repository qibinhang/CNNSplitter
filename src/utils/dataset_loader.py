import copy
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


def get_dataset_loader(dataset_name):
    if dataset_name == 'cifar10':
        load_dataset = _load_cifar10
    elif dataset_name == 'svhn':
        load_dataset = _load_svhn
    elif dataset_name == 'cifar10_svhn':
        load_dataset = _load_inter_dataset
    elif dataset_name == 'svhn_5':
        load_dataset = _load_svhn_5
    else:
        raise ValueError
    return load_dataset


def _load_cifar10(dataset_dir, is_train, labels=None, batch_size=64, num_workers=0, pin_memory=False,
                  is_random=True, part_train=-1):
    """airplane	 automobile	 bird	 cat	 deer	 dog	 frog	 horse	 ship	 truck"""
    if labels is None:
        labels = list(range(10))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    normalize])
    if is_train:
        if is_random:
            transform = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize])

        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform)
        train_targets = np.array(train.targets)
        idx = np.isin(train_targets, labels)
        target_label = train_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        train.targets = trans_label
        train.data = train.data[idx]

        idx = list(range(len(train)))
        np.random.seed(1009)
        np.random.shuffle(idx)
        train_idx = idx[: int(0.8 * len(idx))]
        valid_idx = idx[int(0.8 * len(idx)):]

        if part_train > 0:
            train_idx = train_idx[:part_train]

        train_set = copy.deepcopy(train)
        train_set.targets = [train.targets[idx] for idx in train_idx]
        train_set.data = train.data[train_idx]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=is_random,
                                  num_workers=num_workers, pin_memory=pin_memory)

        valid_set = copy.deepcopy(train)
        valid_set.targets = [train.targets[idx] for idx in valid_idx]
        valid_set.data = train.data[valid_idx]
        valid_loader = DataLoader(valid_set, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, valid_loader

    else:
        test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                            transform=transform)

        test_targets = np.array(test.targets)
        idx = np.isin(test_targets, labels)
        target_label = test_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        test.targets = trans_label
        test.data = test.data[idx]

        test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return test_loader


def _load_svhn(dataset_dir, is_train, labels=None, batch_size=64, num_workers=0, pin_memory=False,
               is_random=True, part_train=-1):
    if labels is None:
        labels = list(range(10))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    if is_train:
        if is_random:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize])
        train = torchvision.datasets.SVHN(root=dataset_dir, split='train', transform=transform)
        train_targets = train.labels
        idx = np.isin(train_targets, labels)
        target_label = train_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        train.labels = np.array(trans_label)
        train.data = train.data[idx]

        idx = list(range(len(train)))
        np.random.seed(1009)
        np.random.shuffle(idx)
        train_idx = idx[: int(0.8 * len(idx))]
        valid_idx = idx[int(0.8 * len(idx)):]

        if part_train > 0:
            train_idx = train_idx[:part_train]

        train_set = copy.deepcopy(train)
        train_set.labels = train.labels[train_idx]
        train_set.data = train.data[train_idx]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=is_random,
                                  num_workers=num_workers, pin_memory=pin_memory)

        valid_set = copy.deepcopy(train)
        valid_set.labels = train.labels[valid_idx]
        valid_set.data = train.data[valid_idx]
        valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, valid_loader
    else:
        test = torchvision.datasets.SVHN(root=dataset_dir, split='test', transform=transform)
        test_targets = test.labels
        idx = np.isin(test_targets, labels)
        target_label = test_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        test.labels = np.array(trans_label)
        test.data = test.data[idx]
        test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return test_loader


def _load_svhn_5(dataset_dir, is_train, labels=list(range(5)), batch_size=64, num_workers=0, pin_memory=False,
                 is_random=True, part_train=-1):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    if is_train:
        if is_random:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize])
        train = torchvision.datasets.SVHN(root=dataset_dir, split='train', transform=transform)
        train_targets = train.labels
        idx = np.isin(train_targets, labels)
        target_label = train_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        train.labels = np.array(trans_label)
        train.data = train.data[idx]

        idx = list(range(len(train)))
        np.random.seed(1009)
        np.random.shuffle(idx)
        train_idx = idx[: int(0.8 * len(idx))]
        valid_idx = idx[int(0.8 * len(idx)):]

        if part_train > 0:
            train_idx = train_idx[:part_train]

        train_set = copy.deepcopy(train)
        train_set.labels = train.labels[train_idx]
        train_set.data = train.data[train_idx]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=is_random,
                                  num_workers=num_workers, pin_memory=pin_memory)

        valid_set = copy.deepcopy(train)
        valid_set.labels = train.labels[valid_idx]
        valid_set.data = train.data[valid_idx]
        valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, valid_loader
    else:
        test = torchvision.datasets.SVHN(root=dataset_dir, split='test', transform=transform)
        test_targets = test.labels
        idx = np.isin(test_targets, labels)
        target_label = test_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        test.labels = np.array(trans_label)
        test.data = test.data[idx]
        test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return test_loader


def _load_inter_dataset(cifar10_dir, cifar10_labels, svhn_dir, svhn_labels,
                        is_train, is_random=True, batch_size=64, num_workers=0, pin_memory=False):
    if is_train:
        cifar10_train, cifar10_val = _load_cifar10(dataset_dir=cifar10_dir, is_train=is_train, labels=cifar10_labels,
                                                   batch_size=batch_size, num_workers=num_workers,
                                                   pin_memory=pin_memory, is_random=is_random)
        svhn_train, svhn_val = _load_svhn(dataset_dir=svhn_dir, is_train=is_train, labels=svhn_labels,
                                          batch_size=batch_size, num_workers=num_workers,
                                          pin_memory=pin_memory, is_random=is_random)

        inter_train_set = _merge_cifar10_svhn(cifar10_train, svhn_train)
        inter_train_loader = DataLoader(inter_train_set, batch_size=batch_size, shuffle=is_random,
                                        num_workers=num_workers, pin_memory=pin_memory)
        inter_valid_set = _merge_cifar10_svhn(cifar10_val, svhn_val)
        inter_valid_loader = DataLoader(inter_valid_set, batch_size=batch_size,
                                        num_workers=num_workers, pin_memory=pin_memory)
        return inter_train_loader, inter_valid_loader
    else:
        cifar10_test = _load_cifar10(dataset_dir=cifar10_dir, is_train=is_train, labels=cifar10_labels,
                                     batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=pin_memory)
        svhn_test = _load_svhn(dataset_dir=svhn_dir, is_train=is_train,
                               labels=svhn_labels, batch_size=batch_size,
                               num_workers=num_workers, pin_memory=pin_memory)
        inter_test_set = _merge_cifar10_svhn(cifar10_test, svhn_test)
        inter_test_loader = DataLoader(inter_test_set, batch_size=batch_size,
                                       num_workers=num_workers, pin_memory=pin_memory)
        return inter_test_loader


def _merge_cifar10_svhn(cifar10_loader, svhn_loader):
    cifar10 = cifar10_loader.dataset
    svhn = svhn_loader.dataset

    cifar10_data, cifar10_labels = cifar10.data, cifar10.targets
    svhn_data, svhn_labels = svhn.data, svhn.labels
    svhn_data = svhn_data.transpose((0, 2, 3, 1))
    svhn_labels = (svhn_labels + max(cifar10_labels) + 1).tolist()

    # merge cifar10_train and svhn_train
    merge_dataset = cifar10
    merge_dataset_labels = cifar10_labels + svhn_labels
    merge_dataset_data = np.concatenate([cifar10_data, svhn_data], axis=0)
    merge_dataset.data = merge_dataset_data
    merge_dataset.targets = merge_dataset_labels
    return merge_dataset
