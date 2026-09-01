from collections import OrderedDict
import json
from pathlib import Path

def group_loss_old(data, exp_name, m_res, m_path_str, f_table_loss):
        coeff_group_uas = {"<1.0": [], '(1, 1)': [], '(1, >=0.8)': [], '(1, <0.8)': []}
        coeff_group_las = {"<1.0": [], '(1, 1)': [], '(1, >=0.8)': [], '(1, <0.8)': []}
        for c_i, c in enumerate(data[f"{exp_name}_coeffs"]):
            uas_value = data[f"{exp_name}_uas"][c_i] if data[f"{exp_name}_uas"][c_i] is not None else 0.0
            las_value = data[f"{exp_name}_las"][c_i] if data[f"{exp_name}_las"][c_i] is not None else 0.0
            if c is not None and c["tok_coeff"] is not None and c["tok_coeff"] == 1.0:
                if c["unlab_coeff"] == 1.0:
                    coeff_group_uas['(1, 1)'].append(uas_value)
                elif c["unlab_coeff"] >= 0.8:
                    coeff_group_uas['(1, >=0.8)'].append(uas_value)
                else:
                    coeff_group_uas['(1, <0.8)'].append(uas_value)
                
                if c["lab_coeff"] == 1.0:
                    coeff_group_las['(1, 1)'].append(las_value)
                elif c["lab_coeff"] >= 0.8:
                    coeff_group_las['(1, >=0.8)'].append(las_value)
                else:
                    coeff_group_las['(1, <0.8)'].append(las_value)
            else:
                coeff_group_uas['<1.0'].append(uas_value)
                coeff_group_las['<1.0'].append(las_value)

        coeff_group_uas = {k: (len(v), round(sum(v) / m_res['all_amount'], 2),
                               round((len(v) - sum(v)) / m_res['all_amount'], 2))
                           for k, v in coeff_group_uas.items()}
        coeff_group_las = {k: (len(v), round(sum(v) / m_res['all_amount'], 2),
                               round((len(v) - sum(v)) / m_res['all_amount'], 2))
                           for k, v in coeff_group_las.items()}

        print(m_path_str + "\n",
            *coeff_group_uas['(1, 1)'][:2], *coeff_group_uas['(1, >=0.8)'],
            *coeff_group_uas['(1, <0.8)'], *coeff_group_uas['<1.0'],
            sep=" & ", end = " \\\\\n", file=f_table_loss)
        
        return coeff_group_uas, coeff_group_las


def group_loss_old2(data, exp_name, m_res, m_path_str, f_table_loss):
        coeff_group_uas = {"(<1, 1)": [], ("(<1, <1)"): [], '(1, 1)': [], '(1, <1)': []}
        coeff_group_las = {"(<1, 1)": [], ("(<1, <1)"): [], '(1, 1)': [], '(1, <1)': []}
        print('(1, 1)', '(1, <1)', '(<1, 1)', '(<1, <1)', file=f_table_loss)
        for c_i, c in enumerate(data[f"{exp_name}_coeffs"]):
            uas_value = data[f"{exp_name}_uas"][c_i] if data[f"{exp_name}_uas"][c_i] is not None else 0.0
            las_value = data[f"{exp_name}_las"][c_i] if data[f"{exp_name}_las"][c_i] is not None else 0.0
            if c is None or c["tok_coeff"] is None:
                tok_coeff = 0.0
                unlab_coeff = 0.0
                lab_coeff = 0.0
            else:
                tok_coeff = c["tok_coeff"]
                unlab_coeff = c["unlab_coeff"]
                lab_coeff = c["lab_coeff"]
            if tok_coeff == 1.0:
                if unlab_coeff == 1.0:
                    coeff_group_uas['(1, 1)'].append(uas_value)
                else:
                    coeff_group_uas['(1, <1)'].append(uas_value)
                
                if lab_coeff == 1.0:
                    coeff_group_las['(1, 1)'].append(las_value)
                else:
                    coeff_group_las['(1, <1)'].append(las_value)
            else:
                if unlab_coeff == 1.0:
                    coeff_group_uas['(<1, 1)'].append(uas_value)
                else:
                    coeff_group_uas['(<1, <1)'].append(uas_value)
                
                if lab_coeff == 1.0:
                    coeff_group_las['(<1, 1)'].append(las_value)
                else:
                    coeff_group_las['(<1, <1)'].append(las_value)

        coeff_group_uas = {k: (len(v), round(sum(v) / m_res['all_amount'], 2),
                               round((len(v) - sum(v)) / m_res['all_amount'], 2))
                           for k, v in coeff_group_uas.items()}
        coeff_group_las = {k: (len(v), round(sum(v) / m_res['all_amount'], 2),
                               round((len(v) - sum(v)) / m_res['all_amount'], 2))
                           for k, v in coeff_group_las.items()}

        print(m_path_str + "\n",
            *coeff_group_uas['(1, 1)'][:2], *coeff_group_uas['(1, <1)'],
            *coeff_group_uas['(<1, 1)'], *coeff_group_uas['(<1, <1)'],
            sep=" & ", end = " \\\\\n", file=f_table_loss)
        
        return coeff_group_uas, coeff_group_las

def group_loss_new(data, exp_name, m_res, m_path_str, f_table_loss_uas, f_table_loss_las):
    '''coeff_group_func_uas  =  [ (lambda tok_coeff, u, _: tok_coeff == 1, '(1, )')
                            , (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and unlab_coeff == 1, '(<1, 1)')
                            , (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and unlab_coeff >= 0.8, '(<1, >=0.8)')
                            , (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and unlab_coeff < 0.8, '(<1, <0.8)')
                           ]
        coeff_group_func_las  =  [ (lambda tok_coeff, u, _: tok_coeff == 1, '(1, )')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and lab_coeff == 1, '(<1, 1)')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and lab_coeff >= 0.8, '(<1, >=0.8)')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and lab_coeff < 0.8, '(<1, <0.8)')
                           ]'''

    coeff_group_func_uas_gr  =  [
                            [ (lambda tok_coeff, u, _: tok_coeff == 1, '(1, )')
                            , (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and unlab_coeff == 1, '(<1, 1)')
                            , (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and unlab_coeff >= 0.8, '(<1, >=0.8)')
                            , (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and unlab_coeff < 0.8, '(<1, <0.8)')
                            ],

                            [  (lambda tok_coeff, unlab_coeff, lab_coeff: tok_coeff == 0.0 and unlab_coeff == 0.0, '(0, 0)')
                            , (lambda tok_coeff, u, _: tok_coeff == 1, '(1, _)')
                            , (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and tok_coeff >= 0.5, '([0.5, 1), _)')
                            , (lambda tok_coeff, unlab_coeff, _: tok_coeff < 0.5, '(<0.5, _)')
                            #, (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and unlab_coeff >= 0.8, '(<1, >=0.8)')
                            #, (lambda tok_coeff, unlab_coeff, _: tok_coeff < 1 and unlab_coeff < 0.8, '(<1, <0.8)')
                            ],

                            [ (lambda tok_coeff, unlab_coeff, _: unlab_coeff == 1, '(_, 1)')
                            , (lambda tok_coeff, unlab_coeff, _: unlab_coeff >= 0.8, '(_, [0.8, 1.0))')
                            , (lambda tok_coeff, unlab_coeff, _: unlab_coeff >= 0.6 and unlab_coeff < 0.8, '(_, [0.6, 0.8))')
                            , (lambda tok_coeff, unlab_coeff, _: unlab_coeff >= 0.4 and unlab_coeff < 0.6, '(_, [0.4, 0.6))')
                            , (lambda tok_coeff, unlab_coeff, _: unlab_coeff >= 0.2 and unlab_coeff < 0.4, '(_, [0.2, 0.4))')
                            , (lambda tok_coeff, unlab_coeff, _: unlab_coeff < 0.2, '(_, [0.0, 0.2))')
                            ],
    ]
    coeff_group_func_las_gr  =  [ 
                            [ (lambda tok_coeff, u, _: tok_coeff == 1, '(1, )')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and lab_coeff == 1, '(<1, 1)')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and lab_coeff >= 0.8, '(<1, >=0.8)')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and lab_coeff < 0.8, '(<1, <0.8)')
                            ],

                            [  (lambda tok_coeff, unlab_coeff, lab_coeff: tok_coeff == 0.0 and lab_coeff == 0.0, '(0, 0)')
                            , (lambda tok_coeff, u, _: tok_coeff == 1, '(1, )')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and tok_coeff >= 0.8, '([0.8, 1), _)')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 0.8 and tok_coeff >= 0.6, '([0.6, 0.8), _)')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 0.6 and tok_coeff >= 0.4, '([0.4, 0.6), _)')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 0.4 and tok_coeff >= 0.2, '([0.2, 0.4), _)')
                            , (lambda tok_coeff, _, lab_coeff: tok_coeff < 0.2, '(<0.2, _)')
                            #, (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and lab_coeff >= , '(<1, >=0.8)')
                            #, (lambda tok_coeff, _, lab_coeff: tok_coeff < 1 and lab_coeff < 0.8, '(<1, <0.8)')
                            ],

                            [ (lambda tok_coeff, _, lab_coeff: lab_coeff == 1, '(_, 1)')
                            , (lambda tok_coeff, _, lab_coeff: lab_coeff >= 0.8, '(_, [0.8, 1.0))')
                            , (lambda tok_coeff, _, lab_coeff: lab_coeff >= 0.6 and lab_coeff < 0.8, '(_, [0.6, 0.8))')
                            , (lambda tok_coeff, _, lab_coeff: lab_coeff >= 0.4 and lab_coeff < 0.6, '(_, [0.4, 0.6))')
                            , (lambda tok_coeff, _, lab_coeff: lab_coeff >= 0.2 and lab_coeff < 0.4, '(_, [0.2, 0.4))')
                            , (lambda tok_coeff, _, lab_coeff: lab_coeff < 0.2, '(_, [0.0, 0.2))')
                            ],
    ]
    assert len(coeff_group_func_uas_gr) == len(coeff_group_func_las_gr)
    for group_i in range(len(coeff_group_func_uas_gr)):
        print("group_i", group_i)
        coeff_group_func_uas = coeff_group_func_uas_gr[group_i]
        coeff_group_func_las = coeff_group_func_las_gr[group_i]

        coeff_group_uas = {x[1]: [] for x in coeff_group_func_uas}
        coeff_group_las = {x[1]: [] for x in coeff_group_func_las}

        for c_i, c in enumerate(data[f"{exp_name}_coeffs"]):
            uas_value = data[f"{exp_name}_uas"][c_i] if data[f"{exp_name}_uas"][c_i] is not None else 0.0
            las_value = data[f"{exp_name}_las"][c_i] if data[f"{exp_name}_las"][c_i] is not None else 0.0
            if c is None or c["tok_coeff"] is None or c["unlab_coeff"] is None or c["lab_coeff"] is None:
                tok_coeff = 0.0
                unlab_coeff = 0.0
                lab_coeff = 0.0
            else:
                tok_coeff = c["tok_coeff"]
                unlab_coeff = c["unlab_coeff"]
                lab_coeff = c["lab_coeff"]
            for check_f, lab in coeff_group_func_uas:
                if check_f(tok_coeff, unlab_coeff, lab_coeff):
                    coeff_group_uas[lab].append(uas_value)
                    break

            for check_f, lab in coeff_group_func_las:
                if check_f(tok_coeff, unlab_coeff, lab_coeff):
                    coeff_group_las[lab].append(las_value)
                    break

        coeff_group_uas = {k: (f"{len(v)} {len(v) / m_res['all_amount']:.2f}", round(sum(v) / m_res['all_amount'], 2),
                               round((len(v) - sum(v)) / m_res['all_amount'], 2))
                           for k, v in coeff_group_uas.items()}
        coeff_group_las = {k: (f"{len(v)} {len(v) / m_res['all_amount']:.2f}", round(sum(v) / m_res['all_amount'], 2),
                               round((len(v) - sum(v)) / m_res['all_amount'], 2))
                           for k, v in coeff_group_las.items()}
        if m_res["uas_all"] != 0:
            print([x[1] for x in coeff_group_func_uas], file=f_table_loss_uas)
            print(m_path_str + "\n",
            f"{m_res['uas_all']:.2f}   ",
            *[v for v in coeff_group_uas.values()],
            sep=" & ", end = " \\\\\n", file=f_table_loss_uas)

        
        if m_res["las_all"] != 0:
            print([x[1] for x in coeff_group_func_las], file=f_table_loss_las)
            print(m_path_str + "\n",
            f"{m_res['las_all']:.2f}   ",
            *[v for v in coeff_group_las.values()],
            sep=" & ", end = " \\\\\n", file=f_table_loss_las)

        print("=====" * 2, file=f_table_loss_uas)
        print("=====" * 2, file=f_table_loss_las)

    return coeff_group_uas, coeff_group_las


def collect_mean_metrics(root_output_dir_path):
    f_table_mean_other = open(f"{root_output_dir_path}/mean_metrics_table_mean_other.txt", 'w')
    f_table_mean_full_prompt = open(f"{root_output_dir_path}/mean_metrics_table_mean_prompt_full.txt", 'w')
    f_table_mean_short_prompt = open(f"{root_output_dir_path}/mean_metrics_table_mean_prompt_short.txt", 'w')

    f_table_mean_orig_lct = open(f"{root_output_dir_path}/mean_metrics_table_mean_ft_orig_lct.txt", 'w')
    f_table_mean_orig_grct = open(f"{root_output_dir_path}/mean_metrics_table_mean_ft_orig_grct.txt", 'w')

    
    f_table_loss_uas_other = open(f"{root_output_dir_path}/mean_metrics_table_loss_uas_other.txt", 'w')
    f_table_loss_uas_full_prompt = open(f"{root_output_dir_path}/mean_metrics_table_loss_uas_prompt_full.txt", 'w')
    f_table_loss_uas_short_prompt = open(f"{root_output_dir_path}/mean_metrics_table_loss_uas_prompt_short.txt", 'w')

    f_table_loss_uas_orig_lct = open(f"{root_output_dir_path}/mean_metrics_table_loss_uas_ft_orig_lct.txt", 'w')
    f_table_loss_uas_orig_grct = open(f"{root_output_dir_path}/mean_metrics_table_loss_uas_ft_orig_grct.txt", 'w')

    f_table_loss_las = open(f"{root_output_dir_path}/mean_metrics_table_loss_las.txt", 'w')

    mean_dict = {}
    p = Path(root_output_dir_path)
    for m_path in list(p.glob(f"**/metrics_*.jsonl")) + list(p.glob(f"**/**/metrics_*.jsonl")): # TODO: фиксация
        m_path_str = str(m_path)
        with open(m_path_str, 'r') as f:
            data = json.load(f)
        exp_name = m_path.stem
        m_res = data[f"{exp_name}_mean"]
        mean_dict[exp_name] = m_res
        mean_dict[exp_name]['mean_tok'] = sum(c["tok_coeff"] if c is not None and c["tok_coeff"] is not None else 0.0
                                           for c in data[f"{exp_name}_coeffs"]) / len(data[f"{exp_name}_coeffs"])

        if "full" in exp_name: # TODO: не унифицировано
            f_table = f_table_mean_full_prompt
            f_table_loss = f_table_loss_uas_full_prompt
        elif "short" in exp_name:
            f_table = f_table_mean_short_prompt
            f_table_loss = f_table_loss_uas_short_prompt
        elif "lct" in exp_name and "original" in exp_name:
            f_table = f_table_mean_orig_lct
            f_table_loss = f_table_loss_uas_orig_lct
        elif "grct" in exp_name and "original" in exp_name:
            f_table = f_table_mean_orig_grct
            f_table_loss = f_table_loss_uas_orig_grct
        else:
            f_table = f_table_mean_other
            f_table_loss = f_table_loss_uas_other
        print(m_path_str + "\n", m_res['wrong_amount'],
            round(m_res['uas_right'], 2), round(m_res['uas_all'], 2),
            round(m_res['las_right'], 2), round(m_res['las_all'], 2),
            sep=" & ", end = " \\\\\n", file=f_table)

        mean_dict[exp_name]['uas_decomp'], mean_dict[exp_name]['las_decomp'] = \
            group_loss_new(data, exp_name, m_res, m_path_str, f_table_loss, f_table_loss_las)

    with open(f"{root_output_dir_path}/mean_metrics.json", 'w') as f:
        json.dump(mean_dict, f, indent=4)
    f_table_mean_orig_grct.close()
    f_table_mean_orig_lct.close()
    f_table_mean_full_prompt.close()
    f_table_mean_short_prompt.close()
    f_table_mean_other.close()

    f_table_loss_uas_orig_grct.close()
    f_table_loss_uas_orig_lct.close()
    f_table_loss_uas_full_prompt.close()
    f_table_loss_uas_short_prompt.close()
    f_table_loss_uas_other.close()
    f_table_loss_las.close()
    #f_table_wrong.close()
    return mean_dict
