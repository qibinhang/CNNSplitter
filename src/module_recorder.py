import argparse
import multiprocessing
import numpy as np
import os
import time
import _pickle as pickle
import itertools
import shutil
import random
from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.module_tools import load_modules, evaluate_ensemble_modules
random.seed(10)


def cal_fitness(generation_idx, explorer_outputs, diff, acc_ratio):
    if number_class == 10:
        n_classification_task = [2, 4, 8, 10]
    elif number_class == 5:
        n_classification_task = [2, 4, 5]
    else:
        raise ValueError
    labels = list(range(number_class))
    sol_compose = []
    for each_class_sol in explorer_outputs:
        tmp = []
        for sol in each_class_sol:
            tmp.append((sol,))
        sol_compose.append(tmp)

    labels_sol_compose = list(zip(labels, sol_compose))
    random.shuffle(labels_sol_compose)
    labels = [item[0] for item in labels_sol_compose]
    sol_compose = [item[1] for item in labels_sol_compose]

    for task in n_classification_task:
        new_sol_compose = []
        for j, i in enumerate(range(0, len(sol_compose), 2)):
            if i + 1 < len(sol_compose):
                ensemble_for_task = list(itertools.product(sol_compose[i], sol_compose[i+1]))
                target_class_label = labels[j*task: (j+1)*task]
                top_ensemble_fitness = _cal_fitness(ensemble_for_task, diff, target_class_label,
                                                    generation_idx=generation_idx, acc_ratio=acc_ratio,
                                                    acc_threshold=acc_thresholds[task],
                                                    min_n_top_acc=50, max_n_top_acc=500)
                top_ensemble_idx = top_ensemble_fitness[:, 0]
                top_ensemble = [ensemble_for_task[int(i)] for i in top_ensemble_idx]
                top_ensemble_unpacking = []
                for item in top_ensemble:
                    tmp = []
                    for its in item:
                        for it in its:
                            tmp.append(it)
                    top_ensemble_unpacking.append(tuple(tmp))
            else:
                top_ensemble_unpacking = sol_compose[i]
            new_sol_compose.append(top_ensemble_unpacking)
        sol_compose = new_sol_compose

    fitness = [[0 for _ in range(configs.num_sol_per_pop)] for _ in range(number_class)]
    for i, compose in enumerate(sol_compose[0]):
        compose_fitness = top_ensemble_fitness[i][1]
        explorer_idx_sol = list(zip(labels, compose))
        for explorer_idx, sol in explorer_idx_sol:
            sol_idx = sol[3]
            last_fitness = fitness[explorer_idx][sol_idx]
            fitness[explorer_idx][sol_idx] = max(last_fitness, compose_fitness)
    fitness = np.array(fitness)
    return fitness, top_ensemble_fitness


def _cal_fitness(ensemble, diff_table, target_class_label, generation_idx, acc_ratio,
                 acc_threshold, min_n_top_acc, max_n_top_acc):
    with multiprocessing.Pool(NUM_CPU, initializer=init_calculating_args,
                              initargs=(target_class_label, diff_table,)) as p:
        results = list(p.map(_calculating, list(enumerate(ensemble))))

    # filter ensembles whose acc is lower than threshold.
    results = np.array(results)
    acc_order = np.argsort(-results[:, 1])
    n_top_acc = int(np.sum(results[:, 1] >= acc_threshold))
    n_top_acc = max(min(n_top_acc, max_n_top_acc), min_n_top_acc)
    top_acc_results = results[acc_order[:n_top_acc]]

    top_acc_ensemble_fitness = acc_ratio * top_acc_results[:, 1] + (1 - alpha) * top_acc_results[:, 2]
    fitness_order = np.argsort(-top_acc_ensemble_fitness)
    top_acc_ensemble_fitness = np.insert(top_acc_results[:, :3], 1, values=top_acc_ensemble_fitness, axis=1)
    sorted_top_acc_ensemble_fitness = top_acc_ensemble_fitness[fitness_order]

    # except 0 generation, the top composes on sub-classification task may not contain the last_generation_best_sol_idx,
    # because the solutions in best compose on 10-classification task
    # may be not the best/good compose on 2/4/8-classification task.
    # Therefore, to make the searching stable, the 'n_top_ensemble' needs to be adjusted dynamically.
    if generation_idx > 0:
        column_is_last_gen_best_sol = results[:, -1]
        column_ensemble_idx = results[:, 0]

        last_gen_best_sol_idx = np.where(column_is_last_gen_best_sol == 1)[0]
        assert len(last_gen_best_sol_idx) == 1
        last_gen_best_sol_ensemble_idx = column_ensemble_idx[last_gen_best_sol_idx]
        if last_gen_best_sol_ensemble_idx not in sorted_top_acc_ensemble_fitness[:, 0]:
            last_best_acc, last_best_diff = results[last_gen_best_sol_idx, 1], results[last_gen_best_sol_idx, 2]
            last_best_fitness = acc_ratio * last_best_acc + (1 - alpha) * last_best_diff
            last_gen_best_sol_data = np.vstack((last_gen_best_sol_ensemble_idx, last_best_fitness,
                                                last_best_acc, last_best_diff)).T
            sorted_top_acc_ensemble_fitness = np.vstack((
                sorted_top_acc_ensemble_fitness, last_gen_best_sol_data
            ))
    return sorted_top_acc_ensemble_fitness


def init_calculating_args(target_class_label, diff_table):
    global g_target_class_label
    global g_diff_table
    g_target_class_label = np.array(target_class_label, dtype=int)
    g_diff_table = diff_table


def _calculating(compose_and_idx):
    compose_idx, compose = compose_and_idx
    sol_idx = []
    pred = []
    label = compose[0][0][1]
    class_label_idx = 0
    is_last_generation_best_sol = False

    # e.g., there are 60^2 'compose', each of which is 4-classification model.
    # each 4-classification model is composed of two 'sub_compose' (i.e., two 2-classification models),
    # and each 'sub_compose', i.e., 2-classification model, is composed of two modules.
    for sub_compose in compose:  # e.g., compose is 4 classification, two sub_compose are 2 classifications.
        for item in sub_compose:  # each 'item' is a module
            sol_idx.append(item[3])
            pred.append(item[0])
            class_label_idx += 1

    if sum(sol_idx) == 0:
        is_last_generation_best_sol = True

    # acc
    ensemble_pred = np.concatenate(pred, axis=1)
    ensemble_pred_label = np.argmax(ensemble_pred, axis=1)
    ensemble_pred_label = g_target_class_label[ensemble_pred_label]

    ensemble_acc = ensemble_pred_label == label
    ensemble_acc = ensemble_acc[np.isin(label, g_target_class_label)]
    ensemble_acc = np.sum(ensemble_acc) / ensemble_acc.shape[0]

    # diff
    ensemble_diff_idx = []
    for i, s_idx in enumerate(sol_idx):
        c_idx = g_target_class_label[i]
        ensemble_diff_idx.append((c_idx, s_idx))

    ensemble_diff = []
    for left, right in itertools.combinations(ensemble_diff_idx, 2):
        if left[0] > right[0]:
            tmp = left
            left = right
            right = tmp
        ensemble_diff.append(g_diff_table[left[0]][right[0]][left[1]][right[1]])

    ensemble_diff = np.mean(ensemble_diff)
    return compose_idx, ensemble_acc, ensemble_diff, is_last_generation_best_sol


def load_explorer_output(gen_idx):
    for explorer_signal in explorer_finish_signal_list:
        while True:
            if os.path.exists(explorer_signal):
                os.remove(explorer_signal)
                break
            else:
                time.sleep(1)

    check_explorer_output(gen_idx)
    outputs = []
    for target_class in range(number_class):
        with open(f'{ga_save_dir}/gen_{gen_idx}_exp_{target_class}_outputs.pkl', 'rb') as f:
            each_output = pickle.load(f)
        outputs.append(each_output)
    return outputs


def check_explorer_output(gen_idx):
    check_results = [_check_explorer_output(tc, gen_idx) for tc in range(number_class)]
    if not all(check_results):
        raise IOError(f"pickle load {ga_save_dir}/gen_{gen_idx}_exp_*_outputs.pkl error!")


def _check_explorer_output(target_class, gen_idx):
    if os.path.exists(f'{ga_save_dir}/gen_{gen_idx}_exp_{target_class}_outputs.pkl'):
        return True
    else:
        return False


def cal_diff_with_multiprocess(explorer_output):
    """use (1 - Jaccard similarity)"""
    all_explorer_sol_active_kernel = []
    for each_explorer_output in explorer_output:
        each_explorer_sol_active_kernel = [each_sol[2] for each_sol in each_explorer_output]
        all_explorer_sol_active_kernel.append(each_explorer_sol_active_kernel)

    explorer_idx_pair_list = []
    for i_idx in range(len(all_explorer_sol_active_kernel)-1):
        for j_idx in range(i_idx + 1, len(all_explorer_sol_active_kernel)):
            explorer_idx_pair_list.append((i_idx, j_idx))

    n_for_process = len(explorer_idx_pair_list) // NUM_CPU + 1
    explorer_idx_pair_list_split = [explorer_idx_pair_list[i: i + n_for_process]
                                    for i in range(0, len(explorer_idx_pair_list), n_for_process)]
    num_cpu = min(NUM_CPU, len(explorer_idx_pair_list_split))
    with multiprocessing.Pool(num_cpu, initializer=init_global_args, initargs=(all_explorer_sol_active_kernel,)) as p:
        results_all_proc = p.map(_cal_diff_with_multiprocess, explorer_idx_pair_list_split)
    results = np.concatenate(results_all_proc, axis=0)

    n_class, n_sol = len(all_explorer_sol_active_kernel), len(all_explorer_sol_active_kernel[0])
    diff = np.zeros((n_class, n_class, n_sol, n_sol), dtype='float32')

    results_idx = results.astype('int16')
    ci_idx, cj_idx, si_idx, sj_idx,  = results_idx[:, 0], results_idx[:, 1], results_idx[:, 2], results_idx[:, 3]
    diff[ci_idx, cj_idx, si_idx, sj_idx] = results[:, 4]
    return diff


def init_global_args(all_explorer_sol_used_kernel):
    global g_all_explorer_sol_used_kernel
    g_all_explorer_sol_used_kernel = all_explorer_sol_used_kernel


def _cal_diff_with_multiprocess(explorer_idx_pair_list):
    results = []
    for explorer_idx_pair in explorer_idx_pair_list:
        i_idx, j_idx = explorer_idx_pair
        diff = _cal_diff(i_idx, j_idx)
        results.append(diff)
    return np.concatenate(results, axis=0)


def _cal_diff(exp_i_idx, exp_j_idx):
    diff = []
    explorer_i = g_all_explorer_sol_used_kernel[exp_i_idx]
    explorer_j = g_all_explorer_sol_used_kernel[exp_j_idx]
    for sol_i_idx, sol_i in enumerate(explorer_i):
        set_i = set(sol_i)
        for sol_j_idx, sol_j in enumerate(explorer_j):
            set_j = set(sol_j)
            sim = len(set_i & set_j) / len(set_i | set_j)
            diff.append((exp_i_idx, exp_j_idx, sol_i_idx, sol_j_idx, 1-sim))
    return np.array(diff, dtype='float32')


def main():
    total_time = 0
    acc_ratio = alpha
    best_generation, best_ensemble, best_fitness, best_acc, best_diff = 0, 0, 0, 0, 0
    early_stop_count = 0
    valid_fitness, valid_acc, valid_diff = 0.0, 0.0, 0.0

    for gen_idx in range(checkpoint, checkpoint + num_generation + 1):  # generation 0 is initial population
        print(f'Generation: {gen_idx}')
        print('-' * 100)
        s_time = time.time()
        outputs = load_explorer_output(gen_idx)
        diff = cal_diff_with_multiprocess(outputs)

        fitness, top_ensemble_fitness = cal_fitness(generation_idx=gen_idx, explorer_outputs=outputs,
                                                    diff=diff, acc_ratio=acc_ratio)

        with open(f'{ga_save_dir}/gen_{gen_idx}_fitness.pkl', 'wb') as f:
            pickle.dump(fitness, f)

        for recorder_finish_signal in recorder_finish_signal_list:
            with open(recorder_finish_signal, 'w') as f:  # the file is just a signal.
                f.write('ok')

        # results on train_dataset
        ensemble = np.argmax(fitness, axis=1)
        _, train_fitness, train_acc, train_diff = top_ensemble_fitness[0]

        # evaluate the acc on valid_dataset
        if gen_idx > 0:
            for i in range(number_class):
                path = f'{configs.ga_save_dir}/gen_{gen_idx}_exp_{i}_pop.pkl'
                while not os.path.exists(path):
                    time.sleep(0.5)

            configs.best_generation = gen_idx
            configs.best_sol_ensemble = ensemble
            modules = load_modules(configs)
            modules = [m[0] for m in modules]
            valid_acc = evaluate_ensemble_modules(modules, valid_dataset)
            valid_diff = train_diff
            valid_fitness = alpha * valid_acc + (1-alpha) * valid_diff  # fix the ratio of validation for early stopping.
            # valid_fitness = 0.9 * valid_acc + 0.1 * valid_diff  # fix the ratio of validation for early stopping.
        time_elapse = time.time() - s_time
        total_time += time_elapse
        print()
        print(f'* ensemble: {ensemble.tolist()}')
        print(f'* fitness [valid]: {valid_fitness * 100:.2f}% (acc: {valid_acc * 100:.2f}%  diff: {valid_diff * 100:.2f}%)')
        print(f'time elapse : {int(time_elapse)} s')
        print()
        acc_ratio = alpha + max(0, gamma - train_acc)

        if valid_fitness > best_fitness:
            best_fitness, best_acc, best_diff = valid_fitness, valid_acc, valid_diff
            best_generation = gen_idx
            best_ensemble = ensemble
            early_stop_count = 0
        elif early_stop:
            early_stop_count += 1
            if early_stop_count == 20:
                print('Early stop.')
                break

    # evaluate the acc on test_dataset
    configs.best_generation = best_generation
    configs.best_sol_ensemble = best_ensemble
    modules = load_modules(configs)
    modules = [m[0] for m in modules]
    valid_acc = evaluate_ensemble_modules(modules, valid_dataset)
    test_acc = evaluate_ensemble_modules(modules, test_dataset)

    time_elapse = total_time // 60
    print(f'Modularity Finished. Time elapse: {time_elapse} min.')
    print(f'During generation {checkpoint} ~ {checkpoint + num_generation}:')
    print(f'* Best generation: {best_generation}')
    print(f'* Best ensemble  : {best_ensemble.tolist()}')
    print(f'* Best fitness   : {best_fitness * 100:.2f}% (acc: {best_acc * 100:.2f}%  diff: {best_diff * 100:.2f}%)')
    print(f'* Valid ACC      : {valid_acc * 100:.2f}%')
    print(f'# Test  ACC      : {test_acc * 100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn', 'svhn_5'])
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--early_stop', action='store_true')
    args = parser.parse_args()
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    checkpoint = args.checkpoint
    early_stop = args.early_stop

    configs = load_configure(model_name, dataset_name)
    num_generation = configs.num_generations
    ga_save_dir = configs.ga_save_dir
    alpha, gamma = configs.alpha, configs.gamma
    explorer_finish_signal_list = configs.explorer_finish_signal_list
    recorder_finish_signal_list = configs.recorder_finish_signal_list

    number_class = 5 if dataset_name == 'svhn_5' else 10

    load_dataset = get_dataset_loader(dataset_name)
    _, valid_dataset = load_dataset(configs.dataset_dir, is_train=True, batch_size=128,
                                    num_workers=1, pin_memory=True, is_random=False)
    test_dataset = load_dataset(configs.dataset_dir, is_train=False, batch_size=128,
                                num_workers=1, pin_memory=True)

    NUM_CPU = configs.num_cpu
    print(f'Multiprocess NUM_CPU: {NUM_CPU}')
    acc_thresholds = configs.acc_thresholds
    print(f'acc_thresholds: {acc_thresholds}\n')

    print(f'=== {model_name}_{dataset_name} ===')
    for key in ['num_generations', 'num_parents_mating', 'num_sol_per_pop', 'keep_parents', 'parent_selection_type',
                'crossover_type', 'mutation_type', 'mutation_percent_genes', 'init_pop_mode',
                'sensitive_layer_idx', 'sensitive_layer_kernel', 'sensitive_layer_group',
                'non_sensitive_layer_idx', 'non_sensitive_layer_kernel', 'non_sensitive_layer_group',
                'sensitive_layer_active_gene_ratio', 'non_sensitive_layer_active_gene_ratio', 'alpha']:
        value = getattr(configs, key)
        print(f'{key} = {value}')
    print()

    if os.path.exists(os.path.dirname(recorder_finish_signal_list[0])):
        shutil.rmtree(os.path.dirname(recorder_finish_signal_list[0]))
    os.makedirs(os.path.dirname(recorder_finish_signal_list[0]))

    main()
