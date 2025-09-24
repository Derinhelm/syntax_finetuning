FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

LABEL Maintenanceer="derinhelm" 

WORKDIR .

RUN pip3 install bitsandbytes transformers peft datasets
