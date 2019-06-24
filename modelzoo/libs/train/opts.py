import argparse
import json


def parse_opts( jsonPath = None ):
    opts = {
        "path": {"root_path": "/home/jimmyyoung/DG_program/model_zoo_torch/modelzoo",
                 "datasest_path": "datasets",
                 "result_path": "results"},
        "model": {"n_classes": 10,
                  "momentum": 0.9,
                  "nesterov": False,
                  "weight_decay": 5e-4,
                  "optimizer": "sgd",
                  },
        "train": {"n_epochs": 15,
                  "learning_rate": 0.1,
                  "batch_size": 128,
                  "no_train": False,
                  "no_val": True,
                  "test": False,
                  "n_threads": 2,
                  "checkpoint": 10,
                  },
        "cuda": {
            "no_cuda": True,
        }
    }
    if jsonPath != None:
        with open(jsonPath, 'r') as result_file:
            opts = json.load(result_file)
    optsJson = json.dumps(opts, sort_keys=True, indent=4, separators=(',', ': '))
    print("----------------config---------------------")
    print(optsJson)
    print("----------------config---------------------")
    return opts
