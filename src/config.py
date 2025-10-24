class ModelConfig:
    def __init__(self, model_config):
        self.is_instruct = model_config.get('is_instruct', False)
        self.model_name = model_config['model_name']
        self.per_device_eval_batch_size = model_config.get('per_device_eval_batch_size', 8)
        self.torch_empty_cache_steps = model_config.get('torch_empty_cache_steps', None)

    def __repr__(self):
        return self.model_name

class DatasetConfig:
    def __init__(self, dataset_config):
        self.train_file_path = dataset_config['train_file_path']
        self.dev_file_path = dataset_config['dev_file_path']
        self.treebank = dataset_config.get('treebank', 'gsd')

    def __repr__(self):
        return f"({self.train_file_path}, {self.dev_file_path}, {self.treebank})"
