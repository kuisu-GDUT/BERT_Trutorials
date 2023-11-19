import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    filename="bert4classify.log"
)

df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv',
                 delimiter='\t', header=None)
logging.info(f"|df info|{df.head()}")
logging.info(f"|df classify: {df[1].value_counts()}")
batch_1 = df[:1000]
logging.info(f"|batch 1 classify: {batch_1[1].value_counts()}")

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (
ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want Trutorial_Transformer instead of distilBERT? Uncomment the following line:
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
logging.info(f"|token info: {tokenizer}")
logging.info(f"|token|special_tokens_map:{tokenizer.special_tokens_map}")
logging.info(f"|token|vocab_size:{tokenizer.vocab_size}")
logging.info(f"|token|special tokens attributes:{tokenizer.SPECIAL_TOKENS_ATTRIBUTES}")

model = model_class.from_pretrained(pretrained_weights)
logging.info(f"|model info: {model}")
logging.info(f"|model|config:\n {model.config}")
logging.info(f"|model|config class:\n {model.config_class}")
logging.info(f"|model|transformer structure:\n {model.transformer}")
logging.info(f"|model|embeddings structure:\n {model.embeddings}")


tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
logging.info(f"|first data: \n{batch_1.head()}| transformer to token: \n{tokenized.head()}")

# padding
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
logging.info(f"|token padding shape: {np.array(padded).shape}|")

# masking
attention_mask = np.where(padded != 0, 1, 0)
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
logging.info(f"|last hidden states:states\n:{last_hidden_states}|")
logging.info(f"|last hidden states shape (batch, src len, hid dim): {last_hidden_states[0].shape}|")

features = last_hidden_states[0][:, 0, :].numpy() # 取第一个字符输出的特征
labels = batch_1[1]

# train and test split, and train the logistic regression
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
logging.info(f"|train shape: {train_features.shape}| test shape: {test_features.shape}")
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

# evaluating logistic model
score = lr_clf.score(test_features, test_labels)
logging.info(f"|Eval logistic classify score: {score}")

from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

scores = cross_val_score(clf, train_features, train_labels)
logging.info("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))