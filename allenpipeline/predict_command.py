"""
The ``predict`` subcommand allows you to make bulk JSON-to-JSON
or dataset to JSON predictions using a trained model and its
:class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ allennlp predict --help
    usage: allennlp predict [-h] [--output-file OUTPUT_FILE]
                            [--weights-file WEIGHTS_FILE]
                            [--batch-size BATCH_SIZE] [--silent]
                            [--cuda-device CUDA_DEVICE] [--use-dataset-reader]
                            [--dataset-reader-choice {train,validation}]
                            [-o OVERRIDES] [--predictor PREDICTOR]
                            [--include-package INCLUDE_PACKAGE]
                            archive_file input_file

    Run the specified model against a JSON-lines input file.

    positional arguments:
      archive_file          the archived model to make predictions with
      input_file            path to or url of the input file

    optional arguments:
      -h, --help            show this help message and exit
      --output-file OUTPUT_FILE
                            path to output file
      --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
      --batch-size BATCH_SIZE
                            The batch size to use for processing
      --silent              do not print output to stdout
      --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
      --use-dataset-reader  Whether to use the dataset reader of the original
                            model to load Instances. The validation dataset reader
                            will be used if it exists, otherwise it will fall back
                            to the train dataset reader. This behavior can be
                            overridden with the --dataset-reader-choice flag.
      --dataset-reader-choice {train,validation}
                            Indicates which model dataset reader to use if the
                            --use-dataset-reader flag is set. (default =
                            validation)
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --predictor PREDICTOR
                            optionally specify a specific predictor to use
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
import time
from typing import List, Iterator, Optional
import argparse
import sys
import json

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of, import_module_and_submodules, prepare_environment
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

from allenpipeline import Annotator


def add_subparser(orig_parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = orig_parser.add_parser("predict", help='Run inference.')

    parser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    parser.add_argument('input_file', type=str, help='path to or url of the input file')

    parser.add_argument('output_file', type=str, help='path to output file')

    parser.add_argument('--weights-file',
                        type=str,
                        help='a path that overrides which weights file to use')

    batch_size = parser.add_mutually_exclusive_group(required=False)
    batch_size.add_argument('--batch-size', type=int, default=32, help='The batch size to use for processing')

    cuda_device = parser.add_mutually_exclusive_group(required=False)
    cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')


    parser.add_argument('-o', '--overrides',
                        type=str,
                        default="",
                        help='a JSON structure used to override the experiment configuration')

    parser.add_argument('--include-package',
                        nargs="+",
                        default=[],
                        help='Name of packages to include')
    parser.set_defaults(func=main)
    return parser


def main(args : argparse.Namespace):

    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    if "annotator" not in config:
        raise ConfigurationError("Key 'annotator' is missing, sorry, cannot perform annotation")

    annotator = Annotator.from_params(config["annotator"])

    if annotator is None:
        raise ConfigurationError("Trained model doesn't have an 'annotator' defined in config file.")
    t1 = time.time()
    annotator.annotate_file(model, args.input_file, args.output_file)
    t2 = time.time()
    print("Predicting took", round(t2-t1,3),"seconds.")

