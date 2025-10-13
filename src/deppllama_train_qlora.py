import gc
import transformers
import time
import yaml

from constants import CUTOFF_LEN, WARMUP_RATIO
from creating_data import creating_data
from creating_model import creating_model # TODO: rename all
from tokenize_functions import InstructTokenizer, BaseTokenizer

import torch


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
    json_train, json_dev = creating_data(parameters)

    #-------------------
    #    LOAD MODEL
    #-------------------
    if parameters.model_config.is_instruct:
        t = InstructTokenizer(parameters)
    else:
        t = BaseTokenizer(parameters)

    # PREPARE DATA
    train_data = ( json_train["train"].shuffle().map(t.generate_and_tokenize_prompt) )
    val_data = ( json_dev["train"].shuffle().map(t.generate_and_tokenize_prompt) )

    original_train_length = len(train_data)
    train_data = remove_example_by_length(train_data, CUTOFF_LEN)

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
        eval_strategy="epoch",
        save_strategy="best",
        output_dir=parameters.output_experiment_path,
        save_total_limit=0,
        group_by_length=parameters.group_by_length,
        load_best_model_at_end=True,
        label_names=["labels"]
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        t.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
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
    trainer.train()
    print(f"Training time:{time.time() - ts}")

    t.tokenizer.save_pretrained(parameters.output_experiment_path)
    model.save_pretrained(parameters.output_experiment_path)
    with open(f"{parameters.output_experiment_path}/config_experiment.yaml", 'w') as file:
        yaml.dump(parameters, file, default_flow_style=False)

    del t
    del model
    for _ in range(3):
        gc.collect() # Сборка мусора для удаления
    torch.cuda.empty_cache()
