from datetime import datetime
import multiprocessing as mp
import time


def run_all_experiments(parallel_config, experiments,
        single_func, parallel_func):
    if not parallel_config:
        for exp in experiments:
            single_func(exp)
    else:
            process_num = 8
            exp_groups = [[] for _ in range(process_num)]
            for i, item in enumerate(experiments):
                exp_groups[i % process_num].append(item)

            mp.set_start_method('spawn', force=True)
             
            start_time = datetime.now().strftime("%D %H:%M:%S").replace("/", "_").replace(":", "_").replace(" ", "_")
            processes = []
            parallel_path = parallel_config["parallel_path"]
            for i in range(process_num):
                p = mp.Process(target=parallel_func,
                    args=(exp_groups[i], i, parallel_path, start_time))
                processes.append(p)
                p.start()
                print(f"Process {i}. is_alive: {p.is_alive()}, params: {p.__dict__}")
                time.sleep(200)
            
            while any(p.is_alive() or p.exitcode != 0 for p in processes):
                for i, p in enumerate(processes):
                    if not p.is_alive() and p.exitcode != 0:
                        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{cur_time}. Error {i} process: {p.exitcode}")
                        processes[i] = mp.Process(target=parallel_func,
                            args=(exp_groups[i], i, parallel_path, start_time))
                        processes[i].start()
                        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{cur_time}. Process {i} is restarted")
                        time.sleep(10)
                        print(f"Process {i}. is_alive: {p.is_alive()}, params: {p.__dict__}")

                time.sleep(30)
