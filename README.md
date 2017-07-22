# Implementation: *Reasoning About Entailment With Neural Attention* [![Build Status](https://travis-ci.com/thundergolfer/reasoning-about-entailment-tensorflow.svg?token=yHGWQ42iK2BPk1FjaUMc&branch=master)](https://travis-ci.com/thundergolfer/reasoning-about-entailment-tensorflow) [![Code Climate](https://codeclimate.com/repos/58fb8909e6e27a027b001f9f/badges/901756218f7b5cb74762/gpa.svg)](https://codeclimate.com/repos/58fb8909e6e27a027b001f9f/feed) [![Test Coverage](https://codeclimate.com/repos/58fb8909e6e27a027b001f9f/badges/901756218f7b5cb74762/coverage.svg)](https://codeclimate.com/repos/58fb8909e6e27a027b001f9f/coverage)

----

Tensorflow implementation of [*Reasoning about Entailment with Neural Attention*](https://arxiv.org/abs/1509.06664), a paper which addresses the problem of [Natural Language Inference](https://nlp.stanford.edu/~manning/talks/SIGIR2016-Deep-Learning-NLI.pdf) with an end-to-end Neural Network architecture. The paper was a collaboration from [Deepmind](https://deepmind.com/), [Oxford University](https://www.cs.ox.ac.uk/activities/machinelearning/), and [University College London](http://mr.cs.ucl.ac.uk/).

----

## Abstract

> While most approaches to automatically recognizing entailment relations have
used classifiers employing hand engineered features derived from complex natural
language processing pipelines, in practice their performance has been only
slightly better than bag-of-word pair classifiers using only lexical similarity. The
only attempt so far to build an end-to-end differentiable neural network for entailment
failed to outperform such a simple similarity classifier. In this paper, we
propose a neural model that reads two sentences to determine entailment using
long short-term memory units. We extend this model with a word-by-word neural
attention mechanism that encourages reasoning over entailments of pairs of words
and phrases. Furthermore, we present a qualitative analysis of attention weights
produced by this model, demonstrating such reasoning capabilities. On a large
entailment dataset this model outperforms the previous best neural model and a
classifier with engineered features by a substantial margin. It is the first generic
end-to-end differentiable system that achieves state-of-the-art accuracy on a textual
entailment dataset.

## Requirements

* `python 2.7` (probably works on Python 3 but untested)
* `tensorflow 1.0.0`
* `gensim` (for loading word embeddings)
* `numpy` and `pandas`

## Installation

#### Prerequisites

First clone this repository with `git clone git@github.com:thundergolfer/reasoning-about-entailment-tensorflow.git` **OR** you can also [fork](https://github.com/thundergolfer/reasoning-about-entailment-tensorflow#fork-destination-box) the repository.

Once cloned, install [Miniconda](https://conda.io/miniconda.html) with `./install_miniconda.sh`. That should succeed outputting `Install complete <INSTALL LOCATION>` in the terminal.

Now the Miniconda has been installed you can install all packages into a nice `conda` virtual environment with `./install.sh`. *[Currently this doesn't support Windows, sorry]*

If everything in the install script went OK, you can enter the created virtual environment with `source ./run_in_environment.sh`.

#### Data & Datasets

* Download the SNLI Dataset ([info](https://nlp.stanford.edu/projects/snli/)): `./scripts/get_snli_dataset.sh`
* Download the pre-trained Word2Vec model ([info](https://en.wikipedia.org/wiki/Word2vec)): `./scripts/get_word2vec.sh`

##### Other Datasets

There are two other datasets that I thought would be interesting to use. First is [*Sentences Involving Compositional Knowledge*](http://clic.cimec.unitn.it/composes/sick.html), a precursor to the *SNLI* dataset.

Second is the [*Multi-Genre NLI Corpus*](https://www.nyu.edu/projects/bowman/multinli/) (*MultiNLI*), which is a successor to the *SNLI* dataset and in some ways superior to it. This new corpus can be combined with the *SNLI* for training. It is also of the same format to the *SNLI*, so the same loading and processing code can be used.

* To get *SICK* run `scripts/get_sick_dataset.sh`
* To get *MultiNLI* run `scripts/get_multiNLI_dateset.sh`

## Usage

Currently training is done by `cd implementation/ && python train.py`

## Results

> Note: These results do not acheive the accuracy results of the paper. Investigating improvements.

![Model accuracy](http://i.imgur.com/wSsEA4r.png)
![Model loss](http://i.imgur.com/Y3OhW8a.png)
![Model performance summary](http://i.imgur.com/dKyefaj.png)

## Training Details

`coming soon`

----------

### References


1. **Tim Rocktäschel, Edward Grefenstette, Karl Moritz Hermann, Tomáš Kočiský, Phil Blunsom**, *[Reasoning about Entailment with Neural Attention](https://arxiv.org/abs/1509.06664)*, 2015.

2. **Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean**, *[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)*, 2013.

3. **Google**, *[Large-Scale Machine Learning on Heterogeneous Systems](http://tensorflow.org/)*, 2015.

4. **Samuel R. Bowman, Gabor Angeli, Christopher Potts, Christopher D. Manning, The Stanford Natural Language Processing Group**, *[A large annotated corpus for learning natural language inference](http://nlp.stanford.edu/projects/snli/)*,  2015.

### Credit

The work done in [borelien/entailment-neural-attention-lstm-tf](https://github.com/borelien/entailment-neural-attention-lstm-tf) and [shyamupa/snli-entailment](https://github.com/shyamupa/snli-entailment) was helpful to me in this project. I hope you find that my work has extended and expanded theirs in interesting and useful ways.

### License

This project is available under the [MIT License](https://choosealicense.com/licenses/mit/)

### Author

Jonathon Belotti ([thundergolfer](https://github.com/thundergolfer)) - [@jonobelotti_IO](https://twitter.com/jonobelotti_IO) - [jonathonbelotti.com](http://jonathonbelotti.com)
