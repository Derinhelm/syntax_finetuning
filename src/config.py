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
        self.test_file_path = dataset_config.get('test_file_path')
        self.conll_test_file_path = dataset_config.get('conll_test_file_path')
        self.treebank_repr = dataset_config.get('treebank_repr', 'grct') # TODO
        self.treebank = dataset_config.get('treebank', 'gsd')

    def __repr__(self):
        return f"({self.train_file_path}, {self.dev_file_path}, {self.treebank})"

class DataRestrictionConfig:
    def __init__(self, restr_config):
        self.index_set = restr_config.get('index_set', None)
        self.index_start = restr_config.get('index_start', None)
        self.index_finish = restr_config.get('index_finish', None)
        assert not (self.index_set is not None and
                    self.index_start is not None) # Не более одного ограничения
        assert not (self.index_set is not None and
                    self.index_finish is not None) # Не более одного ограничения

    def create_index_predicate(self):
        index_predicate = lambda ind: True
        if self.index_set is not None:
            index_predicate = lambda ind: ind in set(self.index_set)
        if self.index_start is not None:
            print(f"{self.index_start=}")
            if self.index_finish is not None:
                index_predicate = lambda ind: (ind >= self.index_start) \
                    and (ind < self.index_finish)
            else:
                index_predicate = lambda ind: ind >= self.index_start
        else:
            if self.index_finish is not None:
                index_predicate = lambda ind: ind < self.index_finish
        return index_predicate
