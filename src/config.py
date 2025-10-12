class ModelConfig:
    def __init__(self, model_config):
        self.is_instruct = model_config.get('is_instruct', False)
        self.model_name = model_config['model_name']

    def __repr__(self):
        return self.model_name

class DatasetConfig:
    def __init__(self, dataset_config):
        self.train_file_path = dataset_config['train_file_path']
        self.dev_file_path = dataset_config['dev_file_path']

    def __repr__(self):
        return f"({self.train_file_path}, {self.dev_file_path})"
