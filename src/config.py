''' Here we will define all the variables which will be used throughout the project'''

# for the dataset
RAW_DATASET = '../input/robert_frost.txt'
PRETRAINED_EMBEDDINGS = '../glove_embeddings/model.txt'
INPUT_SEQ_TRAIN = '../input/input_seq_train.pickle'
OUTPUT_SEQ_TRAIN = '../input/output_seq_train.pickle'

# for dataset generation
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 5000


# for model training
BATCH_SIZE = 128
LR = 0.01
LOSS = 'categorical_crossentropy'
EMBEDDING_DIM = 300



