''' Here we preprocess the text which will be used for training the model'''

# import modules here
import config

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DataPreprocessing:

    # tokenize the input/output
    def tokenize_input(self, dataset):
        '''

        :param dataset: input dataset
        :return:    input sequences, output sequences, tokenizer
        '''
        # these lists will contain input/output texts
        # with <SOS> and <EOS> tags
        input_text = []
        output_text = []

        for line in open(dataset):
            line = line.strip()

            if not line:
                continue
            # adding the sos and eos tags
            input_line = '<SOS>' + line
            output_line = line + '<EOS'

            input_text.append(input_line)
            output_text.append(output_line)

        full_texts = input_text + output_text
        # we need full texts to capture the EOS and SOS tags and assign an index to them
        tokenizer = Tokenizer(num_words=config.MAX_VOCAB_SIZE)
        tokenizer.fit_on_texts(full_texts)
        # we have assigned an index value for each word
        # now we need to assign these indices to the sequences
        input_sequences = tokenizer.texts_to_sequences(input_text)
        output_sequences = tokenizer.texts_to_sequences(output_text)

        return input_sequences, output_sequences, tokenizer
# tokenize the output sequences

# load the glove vectors

# create the embedding matrix

