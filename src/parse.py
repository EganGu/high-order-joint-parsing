import argparse
from utils import ConfParser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--conf')      # option that takes a value
    parser.add_argument('-d', '--device',
                        default=None)  # on/off flag
    parser.add_argument('--spmrl', '-s', action='store_true',
                        help='whether to use spmrl metric')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--lan', '-l', type=str, default="French",
                        help='Which language to use.')
    parser.add_argument('--mode', '-m', type=str, default="train",
                        help='trian, eval or test.')
    parser.add_argument('--data', type=str, help='eval/test data path.')
    parser.add_argument('--data_con', type=str, help='eval/pred data path.')
    parser.add_argument('--data_dep', type=str, help='eval/pred data path.')
    parser.add_argument('--pred', type=str, help='pred save path.')
    parser.add_argument('--path', type=str, help='eval/pred model path.')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs.')
    parser.add_argument('--warmup', type=float, default=0.1, help='training warmup.')
    args = parser.parse_args()
    vars_up = {'device': args.device if args.device is not None else 'cpu',
               'spmrl': args.spmrl, 'lan': args.lan, 'seed': args.seed}
    if args.mode == 'train':
        vars_up.update({'epochs': args.epochs, 'warmup': args.warmup})
        ConfParser(args.conf).train(**vars_up)
    elif args.mode == 'pred':
        pred_args = {'path': args.path, 'data': args.data, 'data_con': args.data_con,
                     'data_dep': args.data_dep, 'pred': args.pred}
        vars_up.update(pred_args)
        ConfParser(args.conf).pred(**vars_up)
    elif args.mode == 'eval':
        eval_args = {'path': args.path, 'data': args.data, 'data_con': args.data_con,
                     'data_dep': args.data_dep}
        vars_up.update(eval_args)
        ConfParser(args.conf).eval(**vars_up)
