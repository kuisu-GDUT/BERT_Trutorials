import datetime
import os.path
import random
import time

import numpy as np
import torch
import pandas as pd
import wandb
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import logging

logging.basicConfig(level=logging.INFO, filename="logs.log")

sweep_config = {
    'method': 'random',  # grid, random
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {

        'learning_rate': {
            'values': [5e-5, 3e-5, 2e-5]
        },
        'batch_size': {
            'values': [16, 32]
        },
        'epochs': {
            'values': [2, 3, 4]
        }
    }
}
sweep_defaults = {
    'learning_rate': 5e-5,
    'batch_size': 32,
    'epochs': 2
}
sweep_id = wandb.sweep(sweep_config)
logging.info(f"|init|sweep_id:{sweep_id}")
wandb.init(config=sweep_defaults)
logging.info(f"|init|configs|{wandb.config.learning_rate}")


class ClassificeCola():
    def __int__(self):
        self.data_path = "./data/raw/in_domain_train.tsv"

    def load_data(self, data_path="./data/raw/in_domain_train.tsv"):
        # load data
        assert (os.path.exists(data_path)), f"{data_path} not exits."
        df = pd.read_csv(data_path, delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])
        logging.info(f"|data head|\n{df.sample(10)}")
        sentences = df.sentence.values
        labels = df.label.values

        return sentences, labels

    def create_token_data(self, sentences: list, labels):
        # tokenizer
        logging.info("|token|loading bert tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        max_len = 0
        # For every sentence...
        for sent in sentences:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = tokenizer.encode(sent, add_special_tokens=True)

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))
        input_ids = []
        attention_masks = []
        logging.info(f"|token|max len:{max_len}")

        # For every sentence...
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        logging.info(f"|token|Original: {sentences[0]}")
        logging.info(f"|token|Token IDs: {input_ids[0]}")
        logging.info(f"|token|label: {labels[0]}")

        # split data
        # Combine the training inputs into a TensorDataset. -->与zip类似
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        logging.info(f"|token|train sample: {train_size}")
        logging.info(f"|token|val sample: {val_size}")
        batch_size = sweep_defaults["batch_size"]
        logging.info(f"|token|batch_size:{batch_size}")
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )
        validation_dataloader = DataLoader(
            val_dataset,
            sampler=RandomSampler(val_dataset),
            batch_size=batch_size
        )
        return train_dataloader, validation_dataloader

    def model(self):
        # create model
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        learning_rate = sweep_defaults["learning_rate"]
        logging.info(f'|model|Learning_rate = {learning_rate}')
        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,
                          eps=1e-8
                          )
        return model, optimizer

    def train(self, train_dataloader, validation_dataloader):
        # start train
        epochs = sweep_defaults["epochs"]
        logging.info(f'|train|epochs =>{epochs}')
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        model, optimizer = self.model()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=total_steps)

        # Set the seed value all over the place to make this reproducible.
        wandb.init()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device("mps")
        logging.info(f"|train|device: {device}")
        model.to(device)

        # print("config ",wandb.config.learning_rate, "\n",wandb.config)
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        # torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        epochs = wandb.config.epochs
        # For each epoch...
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            logging.info('|train|======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            logging.info('|train|Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    logging.info(f'|train|Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.')

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()

                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss, logits = outputs.loss, outputs.logits
                wandb.log({'train_batch_loss': loss.item()})
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            wandb.log({'avg_train_loss': avg_train_loss})

            logging.info("|train|Average training loss: {0:.2f}".format(avg_train_loss))
            logging.info("|train|Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            logging.info("|train|Running Validation...")
            t0 = time.time()
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the computed graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                    (loss, logits) = outputs.loss, outputs.logits

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            logging.info("|trvain|Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)
            wandb.log({'val_accuracy': avg_val_accuracy, 'avg_val_loss': avg_val_loss})
            logging.info("Validation Loss: {0:.2f}".format(avg_val_loss))
            logging.info("Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        logging.info("")
        logging.info("|train|Training complete!")

        logging.info("|train|Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))
        pass

    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def run(self):
        sentences, labels = self.load_data()
        train_dataloader, validation_dataloader = self.create_token_data(sentences, labels)
        self.train(train_dataloader, validation_dataloader)


if __name__ == '__main__':
    cc = ClassificeCola()
    cc.run()
    logging.info("FINIDHED")
