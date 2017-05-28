from __future__ import print_function
import os
import pandas as pd
from gensim.models import KeyedVectors


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
    preprocessed_ds = dict((type_set, {"premises": [], "hypothesis": [], "targets": []}) for type_set in dataset)
    for split in dataset:
        map_targets = {"neutral": 0, "entailment": 1, "contradiction": 2}
        num_ids = len(dataset[split]["targets"])

        for i in range(num_ids):
            try:
                premises_tokens = [word for word in sequence_to_clean_tokens(dataset[type_set]["premises"][i][0])]
                hypothesis_tokens = [word for word in sequence_to_clean_tokens(dataset[type_set]["hypothesis"][i][0])]
                target = map_targets[dataset[type_set]["targets"][i][0]]
            except Exception as e:
                print(e.message)
            else:
                preprocessed_ds[type_set]["premises"].append(premises_tokens)
                preprocessed_ds[type_set]["hypothesis"].append(hypothesis_tokens)
                preprocessed_ds[type_set]["targets"].append(target)

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
