class Parameters:
    def __init__(self, configs):
        self.input_train_path = configs["train_file_path"]
        self.input_dev_path = configs["dev_file_path"]
        self.model_name = configs["model_name"]
        self.treebank = configs.get("treebank", "gsd")
        root_output_dir_path = configs["output_dir_path"]
        clear_model_name = self.model_name.split('/')[-1].replace("-", "_").replace(".", "_")
        self.output_dir_path = f"{root_output_dir_path}/{clear_model_name}_{self.treebank}/"
        self.epochs = configs.get("epochs", 1)
        self.group_by_length = configs.get("group_by_length", False)
        self.disable_qlora = configs.get("disable_qlora", False)
        self.is_instruct = configs.get("is_instruct", False)
        self.batch_size = configs.get("batch_size", 32)
        self.micro_batch_size = configs.get("micro_batch_size", 8)
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size
        self.learning_rate = configs.get("learning_rate", 3e-4)
        self.lora_r = configs.get("lora_r", 8)
        self.lora_alpha = 16
        self.lora_dropout = 0.05
