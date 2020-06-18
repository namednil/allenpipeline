
local num_epochs = 5;
local device = 0;

local word_dim = 100;


local dataset_reader = {
               "type": "tagging_reader",
               "token_indexers": {
                "tokens" : {
                     "type": "single_id"
                     }
                }
           };

local data_iterator = {
        "batch_sampler" : {
            "type": "bucket",
            "batch_size": 16,
            "sorting_keys" : ["tokens"]
        }
    };

local data_writer = {
"type" : "tagging_writer"
};

{
    "dataset_reader": dataset_reader,
    "validation_dataset_reader" : dataset_reader,

    "dataset_writer" : data_writer,

    "validation_command" : {
        "type" : "bash_evaluation_command",
        "command" : "python tagger/eval_script.py {gold_file} {system_output}",
        #"gold_file" : "data/dev.tt", # you can optionally set the gold_file if validation_data_path is not the file that you evaluate against.
        "result_regexes" : {
            "Acc" : [0, "Acc (?P<value>[0-9.]+)"]
        }
    },

   "test_command" : {
        "type" : "bash_evaluation_command",
        "command" : "python tagger/eval_script.py {gold_file} {system_output}",
        "gold_file" : "data/dev.tt",  #you can optionally set the gold_file if test_data_path is not the file that you evaluate against.
        "result_regexes" : {
            "Acc" : [0, "Acc (?P<value>[0-9.]+)"]
        }
    },

    "data_loader": data_iterator,
    "model": {
        "type": "simple_tagger2",
        "encoder" : {
            "type" : "lstm",
            "input_size" : word_dim,
            "hidden_size" : 128,
            "bidirectional" : false,
        },

        "text_field_embedder": {
               "token_embedders": {
                     "tokens" : {
                        "type": "embedding",
                        "embedding_dim": word_dim
                    }
               }
        },

    },
    "train_data_path": "data/dev.tt",
    "validation_data_path": "data/dev.tt",
    "test_data_path" : "data/dev.tt",

    "evaluate_on_test" : true,

    "trainer": {
        "type" : "pipeline",
        "num_epochs": num_epochs,
        "cuda_device": device,
        "optimizer": {
            "type": "adam",
        },
        "checkpointer" : {
            "num_serialized_models_to_keep" : 1,
        },
        "epochs_before_validate" : 1, #if validation is expensive, we can call it slightly later, after first epochs, the model is probably rubbish anyway.

        "external_callbacks" : { # Here we register non-AllenNLP callbacks, that also have access to the annotator.
            "type" : "default", #for some reason, this is necessary.
            "callbacks" : {
                "after_training" : {
                    "type" : "demo-callback"
                }
            }
        }
    },

    # Here we bundle everything that we need to have to make predictions with a trained model:
    "annotator" : {
        "type" : "default",
        "data_loader" : data_iterator,
        "dataset_reader" : dataset_reader,
        "dataset_writer" : data_writer,
    },


}

