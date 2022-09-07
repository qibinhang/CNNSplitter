import argparse
import torch
from ga import GA
from utils.configure_loader import load_configure
from utils.checker import check_dir
from utils.dataset_loader import get_dataset_loader
from utils.model_loader import load_model
from utils.population_initializer import PopulationInitializer
from utils.module_tools import cal_fitness
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_population(checkpoint, init_pop_mode):
    initializer = PopulationInitializer(configs)
    if checkpoint == 0:
        init_pop = initializer.init_population(init_pop_mode)
    else:
        checkpoint_path = f'{ga_save_dir}/gen_{checkpoint}_exp_{target_class}_pop.pkl'
        init_pop = initializer.load_checkpoint(checkpoint_path)
    return init_pop


def fitness_func(solution, solution_idx):
    # module_explorer.py calculate the outputs of solutions,
    # and then module_record.py calculate the fitness of solutions.
    outputs, labels, total_used_kernel_idx = cal_fitness(solution, model, target_class, valid_dataset, configs)
    return outputs, labels, total_used_kernel_idx, solution_idx


def main():
    init_pop = init_population(checkpoint, configs.init_pop_mode)

    explorer = GA(target_class=target_class,
                  save_dir=ga_save_dir,
                  save_checkpoint=True,
                  start_generation=checkpoint,
                  num_generations=configs.num_generations,
                  num_parents_mating=configs.num_parents_mating,
                  fitness_func=fitness_func,
                  initial_population=init_pop,
                  sol_per_pop=configs.num_sol_per_pop,
                  num_genes=init_pop.shape[1],
                  gene_space=[0, 1],
                  explorer_finish_signal=explorer_finish_signal,
                  recorder_finish_signal=recorder_finish_signal,
                  parent_selection_type=configs.parent_selection_type,
                  keep_parents=configs.keep_parents,
                  crossover_type=configs.crossover_type,
                  mutation_type=configs.mutation_type,
                  mutation_percent_genes=configs.mutation_percent_genes)

    explorer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn', 'svhn_5'])
    parser.add_argument('--target_class', choices=[str(i) for i in range(10)])
    parser.add_argument('--checkpoint', type=int, default=0)
    args = parser.parse_args()
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    target_class = int(args.target_class)
    checkpoint = args.checkpoint
    configs = load_configure(model_name, dataset_name)
    trained_model_path = f"{configs.trained_model_dir}/{configs.trained_entire_model_name}"
    ga_save_dir = configs.ga_save_dir
    dataset_dir = configs.dataset_dir
    check_dir(ga_save_dir)
    number_class = 5 if dataset_name == 'svhn_5' else 10
    print(f'=== {model_name}_{dataset_name} ===')
    print(f'Target class {target_class}')

    # prepare kernel indices. sort by importance
    configs.set_sorted_kernel_idx(target_class)
    model = load_model(model_name, num_classes=number_class)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    load_dataset = get_dataset_loader(dataset_name)
    _, valid_dataset = load_dataset(dataset_dir, is_train=True, batch_size=256,
                                    num_workers=3, pin_memory=True, is_random=False)

    check_dir(configs.signal_dir)
    explorer_finish_signal = configs.explorer_finish_signal_list[target_class]
    recorder_finish_signal = configs.recorder_finish_signal_list[target_class]

    main()
