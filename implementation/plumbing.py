from __future__ import print_function
import logging
import os
import pandas as pd
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)

def load_dataset(dataset_dir):
    print("Loading SNLI dataset")
    dataset = {}
    for split in ['train', 'dev', 'test']:
        split_path = os.path.join(dataset_dir, 'snli_1.0_{}.txt'.format(split))
        df = pd.read_csv(split_path, delimiter='\t')
        dataset[split] = {
            "premises": df[["sentence1"]].values,
            "hypothesis": df[["sentence2"]].values,
            "targets": df[["gold_label"]].values
        }

    return dataset


def load_word_embeddings(embeddings_dir):
    print("Loading Word Embeddings")
    return KeyedVectors.load_word2vec_format(embeddings_dir, binary=True)


def dataset_preprocess(dataset):
    logger.info("Tokenisation started...")
    preprocessed_ds = dict((split, {"premises": [], "hypothesis": [], "targets": []}) for split in dataset)
    for split in dataset:
        map_targets = {"neutral": 0, "entailment": 1, "contradiction": 2}
        num_ids = len(dataset[split]["targets"])

        for i in range(num_ids):
            premise, hypothesis  = dataset[split]["premises"][i][0], dataset[split]["hypothesis"][i][0]
	    target = dataset[split]["targets"][i][0]
		
	    if type(premise) is not str or type(hypothesis) is not str:
	        # some examples have "N/A" instead of entry, which gets mapped by pandas to 'nan'
	        continue
		
	    if target not in map_targets:
		# From the SNLI dataset README:
	        # gold_label: This is the label chosen by the majority of annotators. Where no majority exists, this is '-', and the pair should
		#  not be included when evaluating hard classification accuracy.
	        continue

            premises_tokens = [word for word in sequence_to_clean_tokens(premise)]
            hypothesis_tokens = [word for word in sequence_to_clean_tokens(hypothesis)]
            target = map_targets[target]

            preprocessed_ds[split]["premises"].append(premises_tokens)
            preprocessed_ds[split]["hypothesis"].append(hypothesis_tokens)
            preprocessed_ds[split]["targets"].append(target)

    logger.info("Tokenisation done")

    return preprocessed_ds


def sequence_to_clean_tokens(sequence):
    sequence = sequence.lower()

    punctuations = [".", ",", ";", "!", "?", "/", '"', "'", "(", ")", "{", "}", "[", "]", "="]
    for punctuation in punctuations:
        sequence = sequence.replace(punctuation, " {} ".format(punctuation))
    sequence = sequence.replace("  ", " ")
    sequence = sequence.replace("   ", " ")
    sequence = sequence.split(" ")

    unwanted = ["", " ", "  "]

    return [t for t in sequence if t not in unwanted]
