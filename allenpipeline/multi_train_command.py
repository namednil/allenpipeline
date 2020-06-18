
import argparse
import logging
import os
import shutil
from copy import deepcopy
from typing import List, Tuple

import random
import socket

import numpy as np

from allenpipeline import train_command

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',level=logging.INFO) #turn on logging.

try:
    from comet_ml import Experiment
except:
    logging.info("comet_ml package not installed")

from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common import Params



import json
import _jsonnet

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def add_subparser(orig_parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = orig_parser.add_parser("multi-train", help='Run training multiple times and then decide which model to train until completion.')

    parser.add_argument('param_path',
                        type=str,
                        help='path to parameter file describing the model to be trained')

    parser.add_argument('-s', '--serialization-dir',
                        required=True,
                        type=str,
                        help='prefix for directory in which to save the model and its logs')

    parser.add_argument("--retrain", required=True,
                        type=int,help="Number of times to retrain the model")

    parser.add_argument("--epochs", required=True, type=int,
                        help="number of epochs to train before deciding which model is best")

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

    serialization_dir = args.serialization_dir
    num_retrain = args.retrain
    epochs = args.epochs

    if os.path.isdir(serialization_dir):
        if args.force:
            shutil.rmtree(serialization_dir)
        elif len(os.listdir(serialization_dir)) > 1:
            logger.error(serialization_dir + " exists and is not empty")
            raise ValueError(serialization_dir + " exists and is not empty")

    # Identify validation metric
    params = Params.from_file(args.param_path,args.overrides)
    metric_name = params.as_flat_dict().get("trainer.validation_metric","-loss")
    metric_direction = metric_name[0]
    assert metric_direction == "+" or metric_direction == "-"
    metric_name = "best_validation_" + metric_name[1:]

    orig_number_of_epochs = params.as_flat_dict()["trainer.num_epochs"]

    if orig_number_of_epochs <= epochs:
        raise ConfigurationError(f"The number of overall training epochs ({orig_number_of_epochs}) must be larger than those of the 'pre-training' round ({epochs}).")

    # Modify number of epochs
    overrides = json.loads(args.overrides) if args.overrides else dict()
    if not "trainer" in overrides:
        overrides["trainer"] = {}
    overrides["trainer"]["num_epochs"] = epochs

    copied_args = deepcopy(args)
    copied_args.overrides = json.dumps(overrides)
    copied_args.fix = False #make sure that we won't always get the same model.
    copied_args.recover = False
    copied_args.no_archive = True

    val_metrics = []

    #Train some models
    for training_run in range(num_retrain):

        dir = serialization_dir+str(training_run)
        copied_args.serialization_dir = dir
        train_command.main(copied_args)

        #Training done.
        with open(os.path.join(dir, "metrics.json")) as f:
            metrics = json.loads(f.read())
            val_metrics.append(metrics[metric_name])

        #Manipulate the number of epochs to simulate the state as if training was interrupted.
        with open(os.path.join(dir, "config.json")) as f:
            config = json.load(f)
        config["trainer"]["num_epochs"] = orig_number_of_epochs
        with open(os.path.join(dir, "config.json"),"w") as f:
            json.dump(config, f)

    if metric_direction == "-":
        best_epoch, _ = min(enumerate(val_metrics), key=lambda x: x[1])
    else:
        best_epoch, _ = max(enumerate(val_metrics), key=lambda x: x[1])


    print("We got the following validation metrics",val_metrics, "("+metric_name+")")
    print("The best epoch was", best_epoch)
    print("Deleting other models")
    for training_run in range(num_retrain):
        if training_run != best_epoch:
            shutil.rmtree(serialization_dir+str(training_run))

    #Rename best model to original serialization dir
    os.rename(serialization_dir+str(best_epoch), serialization_dir)
    print("Continuing training with epoch", best_epoch, "in", serialization_dir)

    #Train until completion.
    args.recover = True
    args.fix = True
    args.force = False
    train_command.main(args)








