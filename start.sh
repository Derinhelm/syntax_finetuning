#!/bin/bash

pip3 install torch bitsandbytes transformers peft datasets
python3 /src/deppllama_train_qlora.py
