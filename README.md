# allenpipeline

Makes it easier to build models with AllenNLP. 

(Intended) Features:
- simply define a `DatasetWriter` and you're good to go to save your model predictions
- safe model predictions on the dev data
- easy integration of external evaluation tools that receive a system prediction and a gold file
- perform your expensive decoding/parsing on your CPU for all your instances in parallel

## Installation
Clone and install:
```
pip install git+https://github.com/namednil/allenpipeline
```
Make sure you have `allennlp` installed. This was implemented with AllenNLP version `0.9` but other versions might work as well.

## Usage example
Have a look at the code in `tagger` for POS tagging. 

```
bash download_example_data.sh
mkdir -p models/
python -m allenpipeline train tagger/config.jsonnet -s models/testcase -f --include-package tagger
```

## TODO
 - policy for when to run validation with expensive decoding
 - free up GPU memory when doing decoding and no further training
 
 
  