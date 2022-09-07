class GlobalConfigures:
    def __init__(self):
        import os
        root_dir = 'xxx/CNNSplitter'  # the root_dir is 'xx/xx' when the path of this project is xx/xx/CNNSplitter

        self.data_dir = f'{root_dir}/data'
        self.dataset_dir = f'{self.data_dir}/dataset'
        self.trained_entire_model_name = 'entire_model.pth'
        self.kernel_importance_analyzer_mode = ['L1', 'random'][0]
        self.num_cpu = 20
        self.experiments_dir = f'{self.data_dir}/experiments'
