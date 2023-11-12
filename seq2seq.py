import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
import math
import time
from tqdm import tqdm

from models.seq2seq import Seq2Seq, Encoder, Decoder

logging.basicConfig(level=logging.INFO, filename="seq2seq.log")

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Seq2SeqRun():
    def __int__(self):
        pass

    def load_data(self, data_root="/Users/kuisu/Documents/Python/colab-notebooks/pytorch-seq2seq"):
        # load data
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')
        logging.info("|load data|de_core_news_sm and en_core_web_sm load finished!")
        assert os.path.exists(data_root), f"{data_root} not exits"

        def tokenize_de(text):
            """
            Tokenizes German text from a string into a list of strings (tokens) and reverses it
            """
            return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

        def tokenize_en(text):
            """
            Tokenizes English text from a string into a list of strings (tokens)
            """
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = Field(tokenize=tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

        TRG = Field(tokenize=tokenize_en,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)
        train_data, valid_data, test_data = Multi30k.splits(
            exts=('.de', '.en'),
            fields=(SRC, TRG),
            root=data_root
        )
        logging.info(f"|load data\n|train src:{train_data[0].src}\n|train trg:{valid_data[0].trg}")
        logging.info(f"Number of training examples: {len(train_data.examples)}")
        logging.info(f"Number of validation examples: {len(valid_data.examples)}")
        logging.info(f"Number of testing examples: {len(test_data.examples)}")
        logging.info(f"example:\n{vars(train_data.examples[0])}")

        # build vocab
        SRC.build_vocab(train_data, min_freq=2)
        TRG.build_vocab(train_data, min_freq=2)
        logging.info(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
        logging.info(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
        return train_data, valid_data, test_data, SRC, TRG

    def model(self, input_dim, output_dim, device="cpu"):
        # crete model
        INPUT_DIM = input_dim
        OUTPUT_DIM = output_dim
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 2
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

        model = Seq2Seq(enc, dec, device).to(device)

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)

        model.apply(init_weights)
        logging.info(f"model structure:\{model}")
        return model

    def train(self, model, iterator, optimizer, criterion, clip):

        model.train()

        epoch_loss = 0

        for i, batch in tqdm(enumerate(iterator)):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            output = model(src, trg)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(self, model, iterator, criterion):

        model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output = model(src, trg, 0)  # turn off teacher forcing

                # trg = [trg len, batch size]
                # output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]

                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                # trg = [(trg len - 1) * batch size]
                # output = [(trg len - 1) * batch size, output dim]

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    @staticmethod
    def count_parameters(model):
        count_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logging.info(f'The model has {count_params} trainable parameters')
        return count_params

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self):
        # main
        device = torch.device("mps")
        BATCH_SIZE = 128
        train_data, valid_data, test_data, SRC, TRG = self.load_data()

        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            device=device)
        model = self.model(len(SRC.vocab), len(TRG.vocab), device=device)
        optimizer = optim.Adam(model.parameters())
        TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX) # 将pad的损失设置为0

        N_EPOCHS = 2
        CLIP = 1
        best_valid_loss = float('inf')

        for epoch in tqdm(range(N_EPOCHS)):

            start_time = time.time()

            train_loss = self.train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = self.evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'tut1-model.pt')

            logging.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            logging.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            logging.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        model.load_state_dict(torch.load('tut1-model.pt'))

        test_loss = self.evaluate(model, test_iterator, criterion)

        logging.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


if __name__ == '__main__':
    sqsr = Seq2SeqRun()
    sqsr.run()
