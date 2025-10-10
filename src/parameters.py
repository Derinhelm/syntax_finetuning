class Parameters:
    def __init__(self):
        self.train_file_path = None
        self.dev_file_path = None
        self.model_name = None
        self.treebank = "gsd" # TODO: delete default value
        self.root_output_dir_path = None
        self.experiment_number = None
        self.epochs = 1
        self.group_by_length = False
        self.disable_qlora = False
        self.is_instruct = False
        self.batch_size = 32
        self.micro_batch_size = 8
        self.learning_rate = 3e-4
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05

    @property
    def gradient_accumulation_steps(self):
        return self.batch_size // self.micro_batch_size

    @property
    def output_dir_path(self):
        clear_model_name = self.model_name.split('/')[-1].replace("-", "_").replace(".", "_")
        return f"{self.root_output_dir_path}/{clear_model_name}_{self.treebank}/{self.experiment_number}"
