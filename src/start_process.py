import os
import sys
import time
import traceback

import resource

from inference_parser import start_inference_experiment


def start_parallel_inference_experiment(exp_list, process_i, parallel_path, start_time):
    stdout_file = open(f"{parallel_path}/process_{start_time}_{process_i}.out", 'w')
    stderr_file = open(f"{parallel_path}/process_{start_time}_{process_i}.err", 'w')

    # Перенаправляем и stdout, и stderr в файл
    sys.stdout = stdout_file
    sys.stderr = stderr_file

    import torch
    memory_limit = 150 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    cpus = set(range(process_i * 16, (process_i + 1) * 16))
    print(cpus)
    os.sched_setaffinity(0, cpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(process_i)
    time.sleep(10)
    print(f"CPU affinity процесса {process_i}: {os.sched_getaffinity(0)}")
    for i in range(torch.cuda.device_count()):
        print(f" GPU {i} процесса {process_i}: {torch.cuda.get_device_name(i)}")    
    # Мягкий и жесткий лимиты на виртуальную память
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(f"Виртуальная память (RLIMIT_AS):")
    print(f"  Мягкий лимит: {soft / (1024**3):.2f} GB" if soft != resource.RLIM_INFINITY else "  Мягкий лимит: безлимитно")
    print(f"  Жесткий лимит: {hard / (1024**3):.2f} GB" if hard != resource.RLIM_INFINITY else "  Жесткий лимит: безлимитно")

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Информация о памяти
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU memory: {total_memory:.2f} GB")

    # Текущее использование
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Currently allocated: {allocated:.2f} GB")
    print(f"Currently reserved: {reserved:.2f} GB")
    print(f"Free (theoretical): {total_memory - allocated:.2f} GB")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    try:
        for exp in exp_list:
            start_inference_experiment(exp)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Error: {e}")

    stdout_file.close()
    stderr_file.close()
