# The Road Not Taken

Building a Language model using a dataset consisting of poems written by Robert Frost.

The repository has multiple directories, with each serving a different purpose:
- input/: contains the:
    - raw dataset
    - input sequences in tokenized format
    - output sequences in tokenized format
- src/: this directory consists of the source code for the project.
    - config.py: consists of variables which are used all across the code.
    - feature_engg.py: used for preprocessing the data.
    - test_functionalities: using pytest module, i define some sanity checks on the data.
    - model.py: this file contains the code for implementing the model. The train and the inference stage.
    - main.py: this is where all the code comes together. Calling specific functions using commandline arguments.
- plots/: consists of the plots used for observing model loss and accuracy.

## To preprocess the data and generate embeddings, use the following command:
  ```python main.py --preprocess data```
  
  
## To train the model, use the following command:
  ```python main.py --train seq2seq```
  
  
## For generating text 5 lines at a time, use:
  ```python main.py --generate text```

