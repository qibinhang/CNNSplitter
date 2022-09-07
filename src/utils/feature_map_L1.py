import numpy as np
import torch
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_conv_outputs(model, dataset, target_class, confidence_threshold=0.9):
    conv_outputs = []
    with torch.no_grad():
        for data, _ in tqdm(dataset, desc='extracting conv outputs', ncols=100):
            data = data.to(device)
            conv_outputs_for_each_data, pred_confidence = model.extract_conv_outputs(data)
            if pred_confidence[target_class] >= confidence_threshold:
                conv_outputs.append(conv_outputs_for_each_data)
    return conv_outputs


def extract_L1_for_rescnn(model, dataset, target_class, confidence_threshold=0.9):
    feature_map_L1 = []
    with torch.no_grad():
        for data, _ in tqdm(dataset, desc='extracting conv outputs', ncols=100):
            L1_for_each_data, pred_confidence = model.extract_feature_map_L1(data.to(device))
            if pred_confidence[target_class] >= confidence_threshold:
                feature_map_L1.append(L1_for_each_data)
    return feature_map_L1


def calculate_avg_L1_for_rescnn(L1_for_one_class):
    all_layer_L1 = [[] for _ in range(len(L1_for_one_class[0]))]
    for each_data_all_layer_L1 in L1_for_one_class:
        for each_layer_idx, each_layer_L1 in enumerate(each_data_all_layer_L1):
            all_layer_L1[each_layer_idx].append(each_layer_L1)
    avg_L1 = []
    for each_layer_L1 in all_layer_L1:
        each_layer_avg_L1 = np.mean(np.array(each_layer_L1), axis=0)
        avg_L1.append(each_layer_avg_L1.tolist())
    return avg_L1


def calculate_avg_L1_for_simcnn(conv_outputs):
    avg_L1 = []
    # get each kernel's feature map L1s on all class x data
    n_conv = len(conv_outputs[0])
    lfm = [[] for _ in range(n_conv)]
    for outputs_for_each_data in tqdm(conv_outputs, desc=f'calculate L1', ncols=100):
        for i in range(n_conv):
            output = outputs_for_each_data[i]
            L1 = np.sum(output.reshape((output.shape[0], -1)), axis=-1)
            lfm[i].append(L1)

    # get each kernel's average of feature map L1
    for each_layer_L1 in lfm:
        each_layer_L1 = np.array(each_layer_L1)
        each_layer_avg_L1 = np.mean(each_layer_L1, axis=0)
        each_layer_avg_L1 = each_layer_avg_L1.tolist()
        avg_L1.append(each_layer_avg_L1)
    return avg_L1


def get_avg_L1_of_feature_maps(model, load_dataset, dataset_dir, target_classes):
    """
    avg_L1_feature_maps:
    [
        class0 [
            conv_1 [k1, ..., kn],
            ...
            conv_n [k1, ..., kn]
        ]

        ...

        class10 [
            conv_1 [k1, ..., kn],
            ...
            conv_n [k1, ..., kn]
        ]
    ]
    """
    avg_L1_feature_maps = []
    print('generating L1 of feature maps...')
    for target_class in target_classes:
        print(f'class {target_class}:')
        _, valid_dataset = load_dataset(dataset_dir, is_train=True, labels=[target_class],
                                        batch_size=1, is_random=False, num_workers=0, pin_memory=False)

        if model.__class__.__name__ == 'SimCNN':
            conv_outputs = extract_conv_outputs(model, valid_dataset, target_class)
            avg_L1_for_one_class = calculate_avg_L1_for_simcnn(conv_outputs)
            del conv_outputs
        elif model.__class__.__name__ == 'ResCNN':
            L1_for_one_class = extract_L1_for_rescnn(model, valid_dataset, target_class)
            avg_L1_for_one_class = calculate_avg_L1_for_rescnn(L1_for_one_class)
        else:
            raise ValueError

        avg_L1_feature_maps.append(avg_L1_for_one_class)
        print()
    return avg_L1_feature_maps
