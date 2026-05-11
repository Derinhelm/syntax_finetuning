from constants import CUTOFF_LEN

def create_output_model_dataset_path(model_config, dataset_config,
                                     root_output_dir_path):
    clear_model_name = model_config.model_name.split(
        '/')[-1].replace("-", "_").replace(".", "_")
    return f"{root_output_dir_path}/{clear_model_name}_{dataset_config.treebank}"

class Parameters:
    def __init__(self, config_name):
        self.config_name = config_name
        self.dataset_config = None
        self.model_parameters = None
        self.treebank_parameters = None
        self.model_config = None
        self.root_output_dir_path = None
        self.output_model_dataset_path = None
        self.output_experiment_path = None
        self.experiment_number = None
        self.epochs = 1
        self.group_by_length = False
        self.batch_size = 32
        self.micro_batch_size = 8
        self.learning_rate = 3e-4
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.seed = 42
        self.save_epoch_adapters = False
        self.save_steps = None
        self.eval_steps = None
        self.init_lora_weights = True
        self.cutoff_len = CUTOFF_LEN

    @property
    def gradient_accumulation_steps(self):
        return self.batch_size // self.micro_batch_size

    def check_is_none(self):
        return self.model_parameters is None

class InferenceParameters:
    def __init__(self):
        self.logit_parameters = {}
        self.seed = 42
        self.model_name = None
        self.model_parameters = None
        self.treebank_parameters = None
