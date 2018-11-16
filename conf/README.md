# Configuration in Yaml
For each dataset, we need three config files: 
- schema.yaml
- feature.yaml
- train.yaml

## Schema config
This should be consistent with the **data fields**, the order matters.  
Field index start from 1, and the target variable should be named with **`label`**.

### Examples:
```
# Field index: field name
1: xxx
2: xxx
3: xxx

```

## Feature config
Each continuous feature consists 1 attributes **`type`**, 
Each category feature consists 2 attributes **`type`**, **`size`**.
The feature name should be consistent with `schema.yaml`.

### Examples:
```
f1:                 
    type: continuous 
    
f2:                 
    type: category    
    size: 8537                   # distict feature value count  
```  

## Train config
All train config divided into following four part: 
- train 
- model
- runconfig
- distributed

Optional parameters set empty to use default.
Note this configuration set defaults to argparser. Same params can be overrided by using command line.  
For example:   
`python train.py --model_dir ./model_new`

### Examples
```
# Train & Test Parameters
train:
  model_dir: model                  # model base directory            
  train_data: data/criteo/train.csv # train data file path
  dev_data: data/criteo/dev.csv     # validation data file path 
  test_data: data/criteo/test.csv   # test data file path
  batch_size: 256                   # batch size
  train_epochs: 5                   # train epochs
  epochs_per_eval: 1                # evaluation every epochs
  steps_per_eval:                   # evaluation every steps for large dataset, it will override `epochs_per_eval` options. 
  keep_train: 0                     # bool, set true or 1 to keep train from ckpt
  num_samples: 50000000             # train sample size for shuffle buffer size
  checkpoint_path:                  # optional, checkpoint path used for testing  
  skip_lines: 1                     # optional, dataset skip lines
  field_delim: '\t'                 # data field delimeter
  verbose: 1                        # bool, Set 0 for tf log level INFO, 1 for ERROR 

# Model Parameters
model:
  # To use multi mnn model, set nested hidden_units.
  # Examples:
  # hidden_units: [[1024, 12,256], [512,256]] 
  
  hidden_units: [1024, 512, 256]   # hidden_units: List of each hidden layer units, set nested list for Multi MNN. 
  learning_rate: 0.1               # initial learning rate
  lr_decay: false                  # whether to use learning rate decay or not
  lr_decay_steps: 1000             # learning rate decay steps      
  lr_decay_rate: 0.9               # learning rate decay rate
  activation_function: tf.nn.relu  # activation function, must use tf API format
  l1: 0.01                         # optional, l1 regularization lambda
  l2: 0.01                         # optional, l2 regularization lambda
  dropout:                         # optional, dropout rate, 0.1 for drop 10%
  batch_normalization: false       # optional, bool, set true or 1 for use batch normalization
  
# Saving Parameters (Optional)
# Defined in tf.estimator.RunConfig. See details in https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig
runconfig:
  tf_random_seed: 12345
  save_summary_steps: 100           # Defaults to 100
  save_checkpoints_steps:           # Set either save_checkpoints_steps or save_checkpoints_secs
  save_checkpoints_secs: 1500       # Defaults to 600 (10 minutes)
  keep_checkpoint_max: 5            # Defaults to 5
  keep_checkpoint_every_n_hours: 1  # Defaults to 10000
  log_step_count_steps: 100         # Defaults to 100
```
  

