"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp train --help

   usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-f] [-o OVERRIDES]
                         [--file-friendly-logging]
                         [--include-package INCLUDE_PACKAGE]
                         param_path

   Train the specified model on the specified dataset.

   positional arguments:
     param_path            path to parameter file describing the model to be
                           trained

   optional arguments:
     -h, --help            show this help message and exit
     -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the model and its logs
     -r, --recover         recover training from the state in serialization_dir
     -f, --force           overwrite the output directory if it exists
     -o OVERRIDES, --overrides OVERRIDES
                           a JSON structure used to override the experiment
                           configuration
     --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
     --include-package INCLUDE_PACKAGE
                            additional packages to include
"""


import argparse
import logging
import os
from typing import List, Tuple

import random
import socket

import numpy as np
from allennlp.commands.train import TrainModel
from allennlp.common.logging import prepare_global_logging
from allennlp.training import GradientDescentTrainer

from allenpipeline.train_model import TrainPipelineModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO) #turn on logging.

try:
    from comet_ml import Experiment
except:
    logging.info("comet_ml package not installed")

from allennlp.data import DatasetReader


from allenpipeline.PipelineTrainer import PipelineTrainer
#from allenpipeline.PipelineTrainerPieces import PipelineTrainerPieces
from allenpipeline.evaluation_commands import BaseEvaluationCommand
from allenpipeline.utils import merge_dicts, get_hyperparams

from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common import Params
from allennlp.common.util import prepare_environment, dump_metrics, \
    import_module_and_submodules
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.util import create_serialization_dir, evaluate

import json
import _jsonnet

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def add_subparser(orig_parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = orig_parser.add_parser("train", help='Run training.')

    parser.add_argument('param_path',
                        type=str,
                        help='path to parameter file describing the model to be trained')

    parser.add_argument('-s', '--serialization-dir',
                        required=True,
                        type=str,
                        help='directory in which to save the model and its logs')

    parser.add_argument('-r', '--recover',
                        action='store_true',
                        default=False,
                        help='recover training from the state in serialization_dir')

    parser.add_argument('--include-package',
                        nargs="+",
                        default=[],
                        help='Name of packages to include')

    parser.add_argument('--comet',
                        type=str,
                        default=None,
                        help='comet.ml api key, if you want to log with comet.ml')

    parser.add_argument('--workspace',
                        type=str,
                        default=None,
                        help='name of comet.ml workspace')

    parser.add_argument('--project',
                        type=str,
                        default=None,
                        help='name of comet.ml project')

    parser.add_argument('--tags', nargs='+', help='Tags used for comet.ml. Usage: "--tags foo bar" will add two tags')

    parser.add_argument('-f', '--force',
                        action='store_true',
                        required=False,
                        help='overwrite the output directory if it exists')

    parser.add_argument('--fix',
                        action='store_true',
                        required=False,
                        help='Fix seed instead of using a random one.')

    parser.add_argument('-o', '--overrides',
                        type=str,
                        default="",
                        help='a JSON structure used to override the experiment configuration')

    parser.add_argument('--file-friendly-logging',
                        action='store_true',
                        default=False,
                        help='outputs tqdm status on separate lines and slows tqdm refresh rate')

    parser.add_argument("--no-archive",
                        action='store_true',
                        default=False, help="don't tar up model.")

    parser.set_defaults(func=main)
    return parser


def main(args : argparse.Namespace):

    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    params = Params.from_file(args.param_path,args.overrides)

    random_seed, numpy_seed, pytorch_seed = 41,11,302
    if not args.fix:
        random_seed, numpy_seed, pytorch_seed = random.randint(0,999999999),random.randint(0,999999999),random.randint(0,999999999)

    params["random_seed"] = random_seed
    params["numpy_seed"] = numpy_seed
    params["pytorch_seed"] = pytorch_seed
    prepare_environment(params)
    serialization_dir = args.serialization_dir
    create_serialization_dir(params,serialization_dir , args.recover, args.force)
    prepare_global_logging(serialization_dir, args.file_friendly_logging)

    hyperparams = list(get_hyperparams(params.as_dict(infer_type_and_cast=True)))

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    test_file = params.params.get("test_data_path", None)
    validation_data_path = params.get("validation_data_path", None)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    test_command = None
    if evaluate_on_test:
        test_command = BaseEvaluationCommand.from_params(params.pop("test_command"))

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    train_model = TrainPipelineModel.from_params(params=params, serialization_dir=serialization_dir, local_rank=0)

    trainer = train_model.trainer

    if trainer.validation_command is not None:
       trainer.validation_command.maybe_set_gold_file(validation_data_path)


    params.assert_empty('base train command')

    if args.comet is not None:
        experiment = Experiment(api_key=args.comet, workspace=args.workspace, project_name=args.project,parse_args=False,auto_output_logging=None)
        if args.tags:
            experiment.add_tags(args.tags)
        with open(args.param_path) as fil:
            code = "".join(fil.readlines())
        code += "\n\n#=============Full details=============\n\n"
        full_details = _jsonnet.evaluate_file(args.param_path)
        code += full_details
        code += "\n\n#=============IMPORTANT: overwritten options============\n\n"
        code += args.overrides
        experiment.set_code(code, overwrite=True)

        for key,val in hyperparams:
            experiment.log_parameter(key,val)


        experiment.log_parameter("model_directory",serialization_dir)
        experiment.log_parameter("cuda_device",cuda_device)
        experiment.log_parameter("hostname",socket.gethostname())
        experiment.log_parameter("random_seed",random_seed)
        experiment.log_parameter("numpy_seed",numpy_seed)
        experiment.log_parameter("pytorch_seed",pytorch_seed)
    else:
        experiment = None

    try:
        metrics = trainer.train(experiment)
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir)
        raise

    # Evaluate
    if test_file and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights (see pred_test.txt).")
        trainer.annotator.annotate_file(trainer.model, test_file, os.path.join(serialization_dir,"pred_test.txt"))

        if test_command:
            logger.info("Comparing against gold standard.")
            test_command.maybe_set_gold_file(test_file)
            test_metrics = test_command.evaluate(os.path.join(serialization_dir,"pred_test.txt"))
            if experiment:
                with experiment.test():
                    experiment.log_metrics({k : v for k,v in test_metrics.items() if np.isscalar(v)})
            metrics = merge_dicts(metrics, "test",test_metrics)


    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)


    if not args.no_archive:
        # Now tar up results
        archive_model(serialization_dir)
