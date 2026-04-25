import json
import gc
import os
import transformers
from transformers import set_seed
import time
import yaml

from constants import WARMUP_RATIO
from creating_data import creating_data
from creating_model import creating_model # TODO: rename all
from inference_parser import Parser, inference_dataset
from metric_functions.evaluate_one import evaluate_one_experiment, calculate_mean_metrics
from tokenize_functions import InstructTokenizer, BaseTokenizer

from conllu import parse
import torch
import torch._dynamo

from transformers import TrainerCallback

class LoRACallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Сохраняем только адаптеры
        if state.is_world_process_zero:
            model = kwargs['model']
            epoch_dir = os.path.join(args.output_dir, f"epoch_{int(state.epoch)}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            print(f"LoRA adapter saved at {epoch_dir}")

class MemoryOptimizedTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0

    def memory_clean(self):
        # Принудительная очистка CUDA кэша после валидации
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for _ in range(3):
            gc.collect() # Сборка мусора для удаления

    def evaluation_loop(self, *args, **kwargs):
        # Явно используем no_grad для валидации
        with torch.no_grad():
            result = super().evaluation_loop(*args, **kwargs)
        self.memory_clean()
        return result

    def on_step_end(self, args, state, control, **kwargs):
        """
        Вызывается после каждого шага обучения
        """
        # Вызываем родительский метод
        result = super().on_step_end(args, state, control, **kwargs)
        
        self.step_count += 1
        
        # Очистка памяти каждые N шагов
        if self.step_count % 100 == 0: # TODO: as a parameter
            self.memory_clean()
            
        return result

def remove_example_by_length(lst, target_length):
    result = []
    for item in lst:
        if len(item["input_ids"])<target_length:
            result.append(item)
    return result

#============================================
#                   MAIN
#============================================

def conduct_experiment(parameters):
    set_seed(parameters.seed)
    json_train, json_dev = creating_data(parameters)

    #-------------------
    #    LOAD MODEL
    #-------------------
    if parameters.model_config.is_instruct:
        t = InstructTokenizer(parameters.model_config.model_name)
    else:
        t = BaseTokenizer(parameters.model_config.model_name)

    # PREPARE DATA
    train_data = ( json_train["train"].shuffle().map(t.generate_and_tokenize_prompt) )
    val_data = ( json_dev["train"].shuffle().map(t.generate_and_tokenize_prompt) )

    original_train_length = len(train_data)
    train_data = remove_example_by_length(train_data, parameters.cutoff_len)

    if(len(train_data)!=original_train_length):
        print("WARNING:")
        print("original_train_length: " + str(original_train_length))
        print("len(train_data): " + str(len(train_data)))

    model = creating_model(parameters)

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=parameters.micro_batch_size,
        gradient_accumulation_steps=parameters.gradient_accumulation_steps,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=parameters.epochs,
        learning_rate=parameters.learning_rate,
        fp16=True,
        logging_strategy = "steps",
        logging_steps=1,
        optim="paged_adamw_32bit",
        eval_strategy = "steps" if parameters.eval_steps is not None else "epoch",
        eval_steps = parameters.eval_steps if parameters.eval_steps is not None else None,
        save_strategy = "steps" if parameters.save_steps is not None else "no",
        save_steps =  parameters.save_steps if parameters.save_steps is not None else 500, # 500 is default
        save_total_limit=1 if parameters.save_steps is not None else None,
        resume_from_checkpoint=True if parameters.save_steps is not None else False,
        output_dir=parameters.output_experiment_path,
        group_by_length=parameters.group_by_length,
        label_names=["labels"],
        seed=parameters.seed,
        per_device_eval_batch_size=parameters.model_config.per_device_eval_batch_size,
        torch_empty_cache_steps=parameters.model_config.torch_empty_cache_steps,
        ddp_backend=None,
    )
    print(training_arguments)

    data_collator = transformers.DataCollatorForSeq2Seq(
        t.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    if parameters.save_epoch_adapters:
        callbacks = [LoRACallback]
    else:
        callbacks = None

    trainer = MemoryOptimizedTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    model.config.use_cache = False

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if "falcon" in parameters.model_config.model_name:
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    if torch.__version__ >= "2":
        print("YES! I AM 2.0 :-)")
        model = torch.compile(model)
    print("after compile")

    ts = time.time()
    trainer.train(resume_from_checkpoint=parameters.save_steps is not None)
    print(f"Training time:{time.time() - ts}")

    t.tokenizer.save_pretrained(parameters.output_experiment_path)
    model.save_pretrained(parameters.output_experiment_path)
    with open(f"{parameters.output_experiment_path}/config_experiment.yaml", 'w') as file:
        yaml.dump(parameters, file, default_flow_style=False)

    torch.cuda.synchronize()
    del t
    del model
    if torch.__version__ >= "2":
        torch._dynamo.reset()
    for _ in range(3):
        gc.collect() # Сборка мусора для удаления
    torch.cuda.empty_cache()


    if parameters.dataset_config.test_file_path is not None:
        original_model_id = parameters.model_config.model_name
        peft_model_id = parameters.output_experiment_path # TODO
        is_instruct = parameters.model_config.is_instruct
        dataset_repr = parameters.dataset_config.treebank_repr
        seed = parameters.seed
        model_library = "vllm" # TODO: ???
        max_tokens = 3000 # TODO
        os.environ["VLLM_USE_V1"] = "0"
        parser = Parser(original_model_id, peft_model_id, is_instruct,
                            dataset_repr, seed, model_library, max_tokens,
                            dataset_repr, None)
        dataset_path = parameters.dataset_config.test_file_path
        res_name = parameters.output_model_dataset_path.split("/")[-1] # TODO
        output_dir = parameters.output_experiment_path
        result_path = f"{output_dir}/pred_{res_name}.jsonl"
        index_predicate = lambda ind: True
        inference_dataset(parser, dataset_path, result_path, index_predicate)
        parser.clear()
        del parser
        for _ in range(3):
            gc.collect() # Сборка мусора для удаления
        torch.cuda.empty_cache()

        metric_path = f"{output_dir}/metrics_{res_name}.jsonl"
        conll_test_file_path = parameters.dataset_config.conll_test_file_path

        with open(conll_test_file_path, 'r') as file:
            content = file.read()
        gold_sentences = parse(content)

        expir_res_uas, expir_res_las = evaluate_one_experiment(
            gold_sentences, result_path, "jsonl", "difference_easy")
        
        short_filename = metric_path.split("/")[-1].split(".")[0]
        results = {}
        results[f"{short_filename}_uas"] = expir_res_uas
        results[f"{short_filename}_las"] = expir_res_las
        results[f"{short_filename}_mean"] = calculate_mean_metrics(expir_res_uas, expir_res_las)
        with open(metric_path, 'w') as f:
            json.dump(results, f, indent=4) # Using indent for pretty-printing
