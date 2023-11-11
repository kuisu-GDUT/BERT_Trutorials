import unittest
import pandas as pd

class TestBert(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path = "./data/raw/in_domain_train.tsv"

    def test_data(self):
        # show data
        # Load the dataset into a pandas dataframe.
        df = pd.read_csv(self.data_path, delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])

        # Report the number of sentences.
        print('Number of training sentences: {:,}\n'.format(df.shape[0]))

        # Display 10 random rows from the data.
        print(df.sample(10))
