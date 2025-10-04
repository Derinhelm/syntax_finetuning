LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

CUTOFF_LEN = 512
TRIM_LEN = 100000

WARMUP_RATIO = 0.1

