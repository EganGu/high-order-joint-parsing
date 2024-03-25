import os
import re
import numpy as np
import pandas as pd
from collections.abc import Iterable

result_file_type = 'train'  # {train, eval}


def load_metric_names(metric_type):
    if metric_type.startswith('joint'):
        metric_names = ['LOSS', 'DEP-UCM', 'DEP-LCM', 'UAS', 'LAS', 'CON-UCM',
                        'CON-LCM', 'UP', 'UR', 'UF', 'LP', 'LR', 'LF', 'T-UCM', 'T-LCM']
    elif metric_type == 'con':
        metric_names = ['LOSS', 'CON-UCM', 'CON-LCM',
                        'UP', 'UR', 'UF', 'LP', 'LR', 'LF']
    elif metric_type == 'dep':
        metric_names = ['LOSS', 'DEP-UCM', 'DEP-LCM', 'UAS', 'LAS']
    else:
        raise ValueError(f'Unknown metric type: {metric_type}')
    return metric_names


def extract_float(s):
    return np.array(list(map(float, re.findall(r'\d+\.?\d*', s))))


def de_duplication(lst):
    # de-dup and remain the sort in list.
    dd = []
    for e in lst:
        if e not in dd:
            dd.append(e)
    return dd


def get_perf_list(dir):
    return [d for d in os.listdir(dir) if '.' not in d]


def get_perf(p, temp):
    with open(p) as fr:
        lines = fr.readlines()
    for i in range(len(lines)):
        condition = lines[i].endswith('saved\n') if result_file_type == 'train' else lines[i].endswith('Evaluating the data\n')
        if condition:
            # start fetch
            if temp in ['con', 'dep']:
                if result_file_type == 'train':
                    p = {
                        'dev': extract_float(lines[i+1].split(']')[1]),
                        'test': extract_float(lines[i+2].split(']')[1])
                    }
                elif result_file_type == 'eval':
                    p = {'test': extract_float(lines[i+1].split(']')[1])}
            elif temp.startswith('joint'):
                if result_file_type == 'train':
                    p = {
                        'dev': extract_float(lines[i+1].split(']')[1]),
                        'test-satta': extract_float(lines[i+3]),
                        'test-idp': extract_float(lines[i+5])
                    }
                elif result_file_type == 'eval':
                    p = {'test': extract_float(lines[i+1].split(']')[1])}
            else:
                raise ValueError(f'Unknown template type: {temp}')

    return p, load_metric_names(temp)


def get_avg(perf_lst):
    def strong_stat(arr):
        # las_idx = 4
        # lf_idx = -3
        las_idx = -1
        lf_idx = -2
        main_value = arr[:, (las_idx, lf_idx)].sum(-1)
        arr = arr[[x for x in range(len(arr)) if x not in (main_value.argmax(), main_value.argmin())], :]
        # arr = arr[[x for x in range(len(arr)) if x not in (main_value.argmin(),)], :]
        return arr

    assert len(perf_lst)
    collect = {k: [] for k in perf_lst[0].keys()}
    for k in collect.keys():
        for p in perf_lst:
            collect[k].append(p[k])
    collect = {k: np.array(v) for k, v in collect.items()}
    collect = {k: strong_stat(v) for k, v in collect.items()}
    # collect = {k: strong_stat(v) for k, v in collect.items()}
    # collect = {k: strong_stat(v) for k, v in collect.items()}
    # collect = {k: strong_stat(v) for k, v in collect.items()}
    # collect = {k: strong_stat(v) for k, v in collect.items()}
    # collect = {k: strong_stat(v) for k, v in collect.items()}

    avg = {k: np.mean(v, axis=0) for k, v in collect.items()}
    std = {k: np.std(v, axis=0) for k, v in collect.items()}
    # for k, v in collect.items():
    #     print(k)
    #     print(pd.DataFrame(np.array(v)))
    #     print(np.mean(v, axis=0).tolist()[3:])
    #     print(np.std(v, axis=0).tolist()[3:])
    return avg, std


def cal_perf(dir, mode='svj', decode='satta'):

    if isinstance(dir, str):
        if not dir.endswith('/'):
            dir += '/'
        perf_dir = get_perf_list(dir)
    elif isinstance(dir, Iterable):
        perf_dir = [d.rsplit('/', 1)[1] for d in dir]
        dir = dir[0].rsplit('/', 1)[0]+'/'
    else:
        raise ValueError

    # get the model perf unit so that can cal the ave perf
    # default perf log name template: model_name-random_seed
    # sort the perf file
    # m_name, seeds = [x for x in ]
    pf_names = [x.split('-s') for x in perf_dir]
    m_names = de_duplication([s[0] for s in pf_names])
    seeds = de_duplication([s[1] for s in pf_names])

    perf_info = {}
    for m in m_names:
        perfs = []
        for s in seeds:
            if result_file_type == 'train':
                perf_path = dir + m + '-s' + s + '/model.train.log'
            else:
                perf_path = dir + m + '-s' + s + '/model.evaluate.log'
            perf_vals, perf_names = get_perf(perf_path, temp=m)
            perfs.append(perf_vals)
        avg, std = get_avg(perfs)
        perf_info[m] = {
            'avg': avg,
            'std': std
        }

    # print(perf_info)

    if mode == 'svj':
        cols = load_metric_names('joint')
        perf_dev_avg = np.zeros((len(m_names), len(cols))) - 1
        perf_test_avg = np.zeros((len(m_names), len(cols))) - 1
        perf_dev_std = np.zeros((len(m_names), len(cols))) - 1
        perf_test_std = np.zeros((len(m_names), len(cols))) - 1
        for i, m in enumerate(m_names):
            for j, c in enumerate(cols):
                info = perf_info[m]
                local_cols = load_metric_names(m)
                for lci, lc in enumerate(local_cols):
                    if lc == c:
                        if result_file_type == 'train':
                            decode = 'idp' if m.endswith('-lr') else 'satta'
                            if m.startswith('joint'):
                                perf_test_avg[i, j] = info['avg']['test-' + decode][lci]
                                perf_test_std[i, j] = info['std']['test-' + decode][lci]
                            else:
                                perf_test_avg[i, j] = info['avg']['test'][lci]
                                perf_test_std[i, j] = info['std']['test'][lci]
                            perf_dev_avg[i, j] = info['avg']['dev'][lci]
                            perf_dev_std[i, j] = info['std']['dev'][lci]
                        else:
                            perf_test_avg[i, j] = info['avg']['test'][lci]
                            perf_test_std[i, j] = info['std']['test'][lci]
        print('*'*20+'\n'+dir+':\n')
        if result_file_type == 'train':
            print('dev perf:')
            perf_dev_avg = to_df(perf_dev_avg, cols, m_names)
            print(perf_dev_avg)
            latex_format(perf_dev_avg)
        print('\ntest perf:')
        perf_test_avg = to_df(perf_test_avg, cols, m_names)
        perf_test_std = to_df(perf_test_std, cols, m_names)
        print(perf_test_avg)
        print(perf_test_std)
        latex_format(perf_test_avg)
        latex_format(perf_test_std)


def latex_format(df):
    print('\n---------LATEX-------------------\n')
    rec_cols = ['UAS', 'LAS', 'LP', 'LR', 'LF']
    # rec_cols = ['LP', 'LR', 'LF']
    # rec_cols = ['UAS', 'LAS', 'LP', 'LR', 'LF', 'DEP-LCM', 'CON-LCM', 'T-LCM']
    print(df[rec_cols], end='\n\n')
    for i in df.index:
        print(i, '& ' + ' & '.join([str(round(x, 2))
              for x in df.loc[i, rec_cols].values]))


def to_df(arr, c, i):
    df = pd.DataFrame(arr)
    df.columns, df.index = c, i
    return df


if __name__ == '__main__':
    # cal_perf('exp/bert-e20-SPMRL-Swedish')
    cal_perf('exp/bert-e20-SPMRL-Korean')
