import os
import torch
import torch.multiprocessing as mp

from supar.utils import Config
from models import (JointParser, OptCRFDependencyParser,
                    OptCRFConstituencyParser)
from supar.utils.parallel import get_device_count, get_free_port
from supar.cmds.cmd import parse
import configparser


PARSER = {
    'joint': JointParser,
    'con': OptCRFConstituencyParser,
    'dep': OptCRFDependencyParser
}


class ConfParser(object):
    r"""The file-config parser for train/evaluate/predict."""

    def __init__(self, conf_path: str):
        super().__init__()
        self.conf = configparser.RawConfigParser()
        self.conf_path = conf_path
        self.load_conf(conf_path)
        self.core_conf = self.load('Setting')
        self.load_conf(self.core_conf['model_template'])
        self.load_conf(self.core_conf['data_template'])
        # self.conf.read(conf_path, encoding='utf-8')
        self.mapper = self.load('Map')

    def load_conf(self, path, encoding='utf-8'):
        self.conf.read(path, encoding=encoding)

    def load_options(self, conf):
        encoder_conf = self.load(self.core_conf['encoder_type'])
        data_conf = self.load(self.core_conf['data_type'])
        embed_conf = self.load(
            self.core_conf['embed_type']) if conf['embed_type'] is not None else {}
        bert_conf = self.load(
            self.core_conf['bert_type']) if conf['bert_type'] is not None else {}
        expert_opts = {**self.load('Env'), **self.load('Train'), **encoder_conf,
                       **data_conf, **embed_conf, **bert_conf}
        if conf['parser_name'] == 'joint':
            expert_opts.update(self.load('Joint-Option'))
        expert_opts.update(conf)
        return expert_opts

    def map(
        self,
        conf: dict,
        default_maps: dict = {'int': int, 'float': float},
        default_keys: dict = {'True': True, 'False': False, 'None': None}
    ) -> dict:
        for k in list(conf):
            if k in self.mapper:
                converter = default_maps[self.mapper[k]]
                conf[k] = converter(conf[k])
            elif conf[k] in default_keys:
                conf[k] = default_keys[conf[k]]
            elif k == 'parser':
                conf['parser_name'] = conf['parser']
                conf['Parser'] = PARSER[conf.pop(k)]
        return conf

    def load(self, section: str):
        try:
            return dict(self.conf[section].items())
        except KeyError as e:
            print({**self.conf})
            raise AssertionError(f"{self.conf_path} has KeyError: {e}")

    def mkdir(self, conf: dict):
        assert 'path' in conf.keys(), 'Model path is not set in config. '
        if 'PTB' in conf['data_type']:
            data_mark = 'ptb'
        elif 'LCTB' in conf['data_type']:
            data_mark = 'lctb'
        elif 'ZCTB' in conf['data_type']:
            data_mark = 'zctb'
        elif 'SPMRL' in conf['data_type']:
            data_mark = f"SPMRL-{conf['lan']}"
        m_path = f"{conf['encoder_type'][:4].lower()}-e{conf['epochs']}-{data_mark}/{conf['parser_name']}"
        if conf['parser_name'] == 'joint':
            if conf['loss_type'] == 'mm':
                if conf['use_head']:
                    m_path += "2o-" + \
                        ("bi" if conf['head_scorer'] == 'biaffine' else 'tri')
                else:
                    m_path += "1o"
            else:
                m_path += "-mtl"
        if conf['binarize_way'] != 'head':
            m_path += f"-{conf['binarize_way']}"
        conf['path'] += (m_path + f"-s{conf['seed']}/model")
        if conf['checkpoint']:
            assert os.path.exists(
                conf['path']), f"Path {conf['path']} not exists. "
        elif conf['build']:
            pardir = '/'.join(conf['path'].split('/')[:-1])
            assert not os.path.exists(
                pardir), f"Built Model {pardir} is already exist. "
            if not os.path.exists(pardir):
                os.makedirs(pardir)
        else:
            raise ValueError(
                "Train mode without checkpoint or build to attain the model. ")
        return conf

    def modify_data_path(self, conf):
        if 'joint' in conf['path']:
            conf['train_dep'] = conf['train_dep'] % conf['lan']
            conf['dev_dep'] = conf['dev_dep'] % conf['lan']
            conf['test_dep'] = conf['test_dep'] % conf['lan']
            conf['train_con'] = conf['train_con'] % conf['lan']
            conf['dev_con'] = conf['dev_con'] % conf['lan']
            conf['test_con'] = conf['test_con'] % conf['lan']
        else:
            conf['train'] = conf['train'] % conf['lan']
            conf['dev'] = conf['dev'] % conf['lan']
            conf['test'] = conf['test'] % conf['lan']

    def train(self, **args):
        conf = {**self.core_conf, **self.load('Train')}
        conf.update(args)
        conf = self.mkdir(self.map(self.load_options(self.map(conf))))
        if conf['spmrl']:
            self.modify_data_path(conf)
        init(Config(**conf))

    def pred(self, **args):
        conf = {**self.core_conf}
        conf = self.map(self.load_options(self.map(conf)))
        conf.update({**args, 'mode': 'predict'})
        init(Config(**conf))

    def eval(self, **args):
        conf = {**self.core_conf}
        conf = self.map(self.load_options(self.map(conf)))
        conf.update({**args, 'mode': 'evaluate'})
        init(Config(**conf))


def init(args: dict):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # show more bug info but will slow down the training speed.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if get_device_count() > 1:
        os.environ['MASTER_ADDR'] = 'tcp://localhost'
        os.environ['MASTER_PORT'] = get_free_port()
        mp.spawn(parse, args=(args,), nprocs=get_device_count())
    else:
        parse(0 if torch.cuda.is_available() else -1, args)
