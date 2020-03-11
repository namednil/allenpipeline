import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple, Optional

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError

from allenpipeline.utils import merge_dicts, flatten


class BaseEvaluationCommand(ABC, Registrable):
    """
    An evaluation command takes two files (gold and system output) and returns a dictionary with scores.
    """
    def __init__(self):
        self.gold_file = None

    @abstractmethod
    def evaluate(self, system_output:str) -> Dict[str,float]:
        raise NotImplementedError()

    def maybe_set_gold_file(self, gold_file : str) -> None:
        """
        Set gold file if it was None before, otherwise do nothing
        :param gold_file:
        :return:
        """
        if self.gold_file is None:
            if not isinstance(gold_file, str):
                raise ConfigurationError("Attempted to set gold file of evaluation command to something of type "+str(type(gold_file))+" but expected str")
            self.gold_file = gold_file


@BaseEvaluationCommand.register("bash_evaluation_command")
class BashEvaluationCommand(BaseEvaluationCommand):
    """
    An evaluation command that can be configured with jsonnet files.
    Executes a bash command, taps into the output and returns metrics extracted using regular expressions.
    """
    def __init__(self, command : str, result_regexes: Dict[str, Tuple[int, str]], show_output: bool = True, gold_file: Optional[str] = None) -> None:
        """
        Sets up an evaluator.
        :param command: a bash command that will get executed. Use {system_output} and {gold_file} as placeholders.
        :param result_regexes: a dictionary mapping metric names to tuples of line number and regexes how to extract the values of the respective metrics.
            evaluate will return a dictionary where the keys are the metric names and the regexes are used to extract
            the respective values of the metrics in the specified lines. From each regex, we take the group "value". That is, use (?P<value>...) in your regex!
        :param if output of evaluation command should be printed.
        """
        super().__init__()
        if isinstance(gold_file, str) and not os.path.exists(gold_file):
            raise ConfigurationError("Gold file "+gold_file+" doesn't seem to exist")

        self.gold_file = gold_file
        self.command = command
        self.result_regex = result_regexes
        self.show_output = show_output
        for line_number,regex in result_regexes.values():
            assert "(?P<value>" in regex,f"Regex {regex} doesn't seem to contain the group ?P<value>"

    def evaluate(self, system_output: str) -> Dict[str, float]:
        """
        Calls a bash command and extracts metrics.
        :param system_output:
        :param gold_file:
        :return: a dictionary that maps metric names to their values
        """
        with TemporaryDirectory() as direc:
            cmd = self.command.format(system_output=system_output, gold_file=self.gold_file, tmp=direc)
            with subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE) as proc:
                result = bytes.decode(proc.stdout.read())  # output of shell commmand as string
                result_lines = result.split("\n")
                if self.show_output:
                    print(result)
                metrics = dict()
                for metric_name, (line_number,regex) in self.result_regex.items():
                    m = re.search(regex, result_lines[line_number])
                    if m:
                        val = float(m.group("value"))
                        metrics[metric_name] = val
                if self.show_output:
                    print(metrics)
                return metrics


@BaseEvaluationCommand.register("json_evaluation_command")
class JsonEvaluationCommand(BaseEvaluationCommand):
    """
    An evaluation command that can be configured with jsonnet files.
    Executes a bash command, taps into the output and returns metrics extracted using json.
    """
    def __init__(self, commands: List[List[str]], show_output: bool = True, gold_file: Optional[str] = None) -> None:
        """
        Sets up an evaluator.
        :param commands: a list of pairs of (metric_prefix, command) that will get executed. Use {system_output} and {gold_file} and {tmp} as placeholders.
        {tmp} points to a private temporary directory. if metric_prefix is the empty string, no metric will be saved.
        :param if output of evaluation command should be printed.
        """
        super().__init__()
        if isinstance(gold_file, str) and not os.path.exists(gold_file):
            raise ConfigurationError("Gold file "+gold_file+" doesn't seem to exist")

        self.gold_file = gold_file
        self.commands = commands
        for cmd in self.commands:
            assert len(cmd) == 2, "Should get a tuple of [metric_prefix, command] but got "+str(cmd)
        self.show_output = show_output

    def evaluate(self, system_output: str) -> Dict[str, float]:
        """
        Calls the bash commands and extracts metrics for
        :param system_output:
        :param gold_file:
        :return: a dictionary that maps metric names to their values
        """
        metrics : Dict[str,float] = dict()
        with TemporaryDirectory() as direc:
            for prefix,cmd in self.commands:
                cmd = cmd.format(system_output=system_output, gold_file=self.gold_file, tmp=direc)
                with subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE) as proc:
                    result = bytes.decode(proc.stdout.read())  # output of shell commmand as string
                    if self.show_output:
                        print(result)
                    if prefix:
                        try:
                            result_json = json.loads(result)
                            metrics = merge_dicts(metrics, prefix, flatten(result_json))
                        except json.decoder.JSONDecodeError: #probably not intended for us
                            if self.show_output:
                                print("<-- not well-formed json, ignoring")

        if self.show_output:
            print(metrics)
        return metrics


