import argparse
import copy
import torch
from tqdm import tqdm
from utils.checker import check_dir
from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.model_loader import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, save_path):
    global lr
    loss_func = torch.nn.CrossEntropyLoss().to(device)
    best_acc, best_epoch = 0.0, 0
    best_model = None
    early_stop_count = 0
    optimization = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(n_epochs):
        print(f'epoch {epoch}')
        print('-'*80)

        # adjust_learning_rate
        if epoch in lr_schedule:
            lr *= gamma
            for param_group in optimization.param_groups:
                param_group['lr'] = lr

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

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop and early_stop_count == 10:
                print(f'Early Stop.\n')
                break
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
    dataset_dir = configs.dataset_dir
    check_dir(configs.trained_model_dir)
    save_path = f'{configs.trained_model_dir}/{configs.trained_entire_model_name}'

    load_dataset = get_dataset_loader(dataset_name)
    model = load_model(model_name, num_classes=configs.num_classes).to(device)
    print(model)

    train_loader, val_loader = load_dataset(dataset_dir, is_train=True,
                                            batch_size=batch_size, num_workers=1, pin_memory=True)
    test_loader = load_dataset(dataset_dir, is_train=False,
                               batch_size=batch_size, num_workers=1, pin_memory=True)

    if evaluation:
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        test(model, test_loader)
    else:
        model = train(model, train_loader, val_loader, save_path)
        model.eval()
        test(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn', 'svhn_5'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--schedule', type=str, default='60,120', help='Decrease learning rate at these epochs.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    args = parser.parse_args()
    print(args)
    print()

    model_name = args.model
    dataset_name = args.dataset
    lr = args.lr
    batch_size = args.batch_size
    gamma = args.gamma
    lr_schedule = [int(s) for s in args.schedule.split(',')]
    n_epochs = args.epochs
    early_stop = args.early_stop
    evaluation = args.evaluation
    main()
