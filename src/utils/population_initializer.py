import numpy as np
import _pickle as pickle
from numpy import random
np.random.seed(12)


class PopulationInitializer:
    def __init__(self, configs):
        super(PopulationInitializer, self).__init__()
        self.configs = configs

    def init_population(self, init_pop_mode):
        if init_pop_mode == 'random':
            init_pop = self.random_init()
        else:
            init_pop = self.heuristic_init()
        return init_pop

    def random_init(self):
        init_pop = []
        len_chromosome = sum(self.configs.sensitive_layer_group + self.configs.non_sensitive_layer_group)
        for i in range(self.configs.num_sol_per_pop):
            chromosome = np.random.randint(low=0, high=2, size=len_chromosome)
            init_pop.append(chromosome)
        init_pop = np.array(init_pop)
        return init_pop

    def heuristic_init(self):
        init_pop = []
        sensitive_layer_groups = self.configs.sensitive_layer_group
        sensitive_layer_active_gene_ratio = self.configs.sensitive_layer_active_gene_ratio
        sensitive_layer_idx = self.configs.sensitive_layer_idx

        non_sensitive_layer_groups = self.configs.non_sensitive_layer_group
        non_sensitive_layer_active_gene_ratio = self.configs.non_sensitive_layer_active_gene_ratio

        n_sol_per_pop = self.configs.num_sol_per_pop
        for i in range(n_sol_per_pop):
            sensitive_part = self.gen_part_chromosome(sensitive_layer_groups, sensitive_layer_active_gene_ratio)
            non_sensitive_part = self.gen_part_chromosome(non_sensitive_layer_groups,
                                                          non_sensitive_layer_active_gene_ratio)
            # concatenate according to sensitive_layer_idx
            chromosome = []
            sensitive_point, non_sensitive_point = 0, 0
            for idx in range(len(sensitive_part) + len(non_sensitive_part)):
                if idx in sensitive_layer_idx:
                    chromosome.append(sensitive_part[sensitive_point])
                    sensitive_point += 1
                else:
                    chromosome.append(non_sensitive_part[non_sensitive_point])
                    non_sensitive_point += 1
            chromosome = np.concatenate(chromosome, axis=0)
            assert len(chromosome) == sum(sensitive_layer_groups + non_sensitive_layer_groups)
            init_pop.append(chromosome)
        init_pop = np.array(init_pop)
        return init_pop

    def gen_part_chromosome(self, groups, active_gene_ratio):
        len_part_chromosome = sum(groups)
        start_point = 0
        part_chromosome = np.zeros(len_part_chromosome)
        for ng in groups:
            n_active_gene = int(ng * random.choice(active_gene_ratio))
            base = np.array(list(range(ng - 1, -1, -1))) / 10  # /10 to make the prob more smooth
            prob = np.exp(base) / np.sum(np.exp(base), axis=0)
            active_gene_idx = random.choice(list(range(ng)), size=n_active_gene, replace=False, p=prob)
            active_gene_idx = active_gene_idx + start_point
            part_chromosome[active_gene_idx] = 1
            start_point += ng

        # not like SimCNN, the sensitive layer indices in ResCNN are not continuous, so splitting part_chromosome
        split_part_chromosome = []
        start_point = 0
        for ng in groups:
            split_part_chromosome.append(part_chromosome[start_point: start_point + ng])
            start_point += ng
        assert (np.concatenate(split_part_chromosome, axis=0) == part_chromosome).all()
        return split_part_chromosome

    def load_checkpoint(self, checkpoint_path):
        """
        :param checkpoint_path: the name of file which records the last population.
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint_pop = pickle.load(f)
        return checkpoint_pop
