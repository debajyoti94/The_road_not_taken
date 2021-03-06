""" Here we will define all the variables which
 will be used throughout the project"""

# for the dataset
RAW_DATASET = "../input/robert_frost.txt"
PRETRAINED_EMBEDDINGS = "../glove_embeddings/model.txt"
INPUT_SEQ_TRAIN = "../input/input_seq_train.pickle"
OUTPUT_SEQ_TRAIN = "../input/output_seq_train.pickle"

# for dataset generation
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 5000


# for model training
TOKENIZER = "../models/tokenzier.pickle"
ACTUAL_SEQ_LENGTH = "../models/actual_sequence_len.pickle"
EMBEDDING_MATRIX = "../models/embedding_matrix.pickle"
ACTUAL_VOCAB_SIZE = "../models/actual_vocab_size.pickle"
WORD_VECTOR_DICT = "../models/word_vector_dict.pickle"

ENCODER_MODEL = "../models/encoder_model.h5"
SAMPLING_MODEL = "../models/sampling_model.h5"

BATCH_SIZE = 128
NUM_EPOCHS = 100
HIDDEN_DIM = 100
LR = 0.01
LOSS_FN = "categorical_crossentropy"
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
