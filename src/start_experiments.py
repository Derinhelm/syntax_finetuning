from datetime import datetime
import multiprocessing as mp
import time

from start_process import start_parallel_experiment

def run_all_experiments(parallel_config, ft_experiments, inf_experiments,
        function_executor):
    if not parallel_config:
        for exp in ft_experiments:
            function_executor(exp, inf_experiments)
    else:
            process_num = 8
            exp_groups = [[] for _ in range(process_num)]
            inf_groups = [[] for _ in range(process_num)]
            for i, item in enumerate(ft_experiments):
                exp_groups[i % process_num].append(item)
                
            if not(len(ft_experiments) == 1 and ft_experiments[0].check_none()):
                for i, _ in enumerate(inf_groups):
                    inf_groups.append(inf_experiments)
            else:
                for i, item in enumerate(inf_experiments):
                    inf_groups[i % process_num].append(item)

            mp.set_start_method('spawn', force=True)
             
            start_time = datetime.now().strftime("%D %H:%M:%S").replace("/", "_").replace(":", "_").replace(" ", "_")
            processes = []
            parallel_path = parallel_config["parallel_path"]
            for i in range(process_num):
                p = mp.Process(target=start_parallel_experiment,
                    args=(inf_groups[i], exp_groups[i], i,
                    parallel_path, start_time, function_executor))
                processes.append(p)
                p.start()
                print(f"Process {i}. is_alive: {p.is_alive()}, params: {p.__dict__}")
                time.sleep(200)
            
            while any(p.is_alive() or p.exitcode != 0 for p in processes):
                for i, p in enumerate(processes):
                    if not p.is_alive() and p.exitcode != 0:
                        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{cur_time}. Error {i} process: {p.exitcode}")
                        processes[i] = mp.Process(target=start_parallel_experiment,
                            args=(inf_experiments, exp_groups[i], i,
                            parallel_path, start_time, function_executor))
                        processes[i].start()
                        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{cur_time}. Process {i} is restarted")
                        time.sleep(10)
                        print(f"Process {i}. is_alive: {p.is_alive()}, params: {p.__dict__}")

                time.sleep(30)
