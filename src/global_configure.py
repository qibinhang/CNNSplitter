class GlobalConfigures:
    def __init__(self):
        import os
        root_dir = '/Users/qibinhang/Documents/Code/NNModularity/CNNSplitter_ASE22'  # the root_dir is 'xx/xx' when the path of this project is xx/xx/CNNSplitter

        # TEST
        self.data_dir = f'{root_dir}/data'
        #

        # if os.path.exists(f'{root_dir}/CNNSplitter/data'):
        #     self.data_dir = f'{root_dir}/CNNSplitter/data'
        # else:
        #     raise ValueError(f'{root_dir}/CNNSplitter/data does not exist.')
        self.dataset_dir = f'{self.data_dir}/dataset'
        self.trained_entire_model_name = 'entire_model.pth'
        self.kernel_importance_analyzer_mode = ['L1', 'random'][0]
        self.num_cpu = 20
        self.experiments_dir = f'{self.data_dir}/experiments'
