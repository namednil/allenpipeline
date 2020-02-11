import argparse

from . import train_command
from . import predict_command

parser = argparse.ArgumentParser(description='Run allenpipeline')
subparsers = parser.add_subparsers(title='Commands', metavar='')

train_command.add_subparser(subparsers)
predict_command.add_subparser(subparsers)

args = parser.parse_args()

if "func" in dir(args):
    args.func(args)
else:
    parser.print_help()
