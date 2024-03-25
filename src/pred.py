import argparse
from utils import ConfParser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('--mode', default='pred')
    parser.add_argument('--path')      # option that takes a value
    # option that takes a value
    parser.add_argument('--data_dep', default=None)
    parser.add_argument('--data_con', default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--pred')
    # decoding
    parser.add_argument('--decode', default='satta')
    parser.add_argument('-c', '--conf',
                        default=None)  # on/off flag
    parser.add_argument('-d', '--device',
                        default=None)  # on/off flag
    parser.add_argument('-s', '--seed', type=int,
                        default=0)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--n-buckets', type=int, default=32)
    args = parser.parse_args()
    if args.mode == 'pred':
        vars_up = {'device': args.device, 'data_dep': args.data_dep, 'data_con': args.data_con, 'data': args.data,
                   'pred': args.pred, 'path': args.path, 'proj': True, 'decoding': args.decode, 'seed': args.seed,
                   'batch_size': args.batch_size, 'buckets': args.n_buckets}
        ConfParser(args.conf).pred(**vars_up)
    elif args.mode == 'eval':
        vars_up = {'device': args.device, 'data_dep': args.data_dep, 'data_con': args.data_con, 'data': args.data,
                   'path': args.path, 'proj': True, 'decoding': args.decode, 'seed': args.seed, 'batch_size': args.batch_size,
                   'buckets': args.n_buckets}
        ConfParser(args.conf).eval(**vars_up)
