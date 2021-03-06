# allenpipeline

Makes it easier to build a pipeline with AllenNLP that annotates files.

Features:
- comet.ml integration
- simply define a `DatasetWriter` and you're good to go to save your model predictions (in the right order!)
- safe your model's predictions on the dev data every epoch
- easy integration of external evaluation tools that receive a system prediction and a gold file
- have your scores for your expensive decoding/parsing problem pre-computed on the GPU and then perform the parsing in parallel on the CPU.

## Installation
Clone and install:
```
pip install git+https://github.com/namednil/allenpipeline
```
Make sure you have `allennlp` installed. This was implemented for AllenNLP version `1.0`. Checkout the `0.9` branch for a version that was developed for AllenNLP `0.9` but that might also worker for older ones:
```
pip install git+https://github.com/namednil/allenpipeline/@0.9
```

## Usage example
Have a look at the code in `tagger` for POS tagging. 

```
mkdir -p data/
bash download_example_data.sh
mkdir -p models/
python -m allenpipeline train tagger/config.jsonnet -s models/testcase -f --include-package tagger
```

## Known issues
 - doesn't free up GPU memory when doing decoding and no further training
 - comet.ml packages is required, should be optional
 - distributed training not supported.
  
