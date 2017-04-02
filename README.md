# Implementation: *Reasoning About Entailment With Neural Attention* [![Build Status](https://travis-ci.com/thundergolfer/reasoning-about-entailment-tensorflow.svg?token=yHGWQ42iK2BPk1FjaUMc&branch=master)](https://travis-ci.com/thundergolfer/reasoning-about-entailment-tensorflow)

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

`coming soon`

## Installation 

First clone this repository with `git clone git@github.com:thundergolfer/reasoning-about-entailment-tensorflow.git` **OR** you can also [fork](https://github.com/thundergolfer/reasoning-about-entailment-tensorflow#fork-destination-box) the repository.

Once cloned, install [Miniconda](https://conda.io/miniconda.html) with `./install_miniconda.sh`. That should succeed outputting `Install complete <INSTALL LOCATION>` in the terminal. 

Now the Miniconda has been installed you can install all packages into a nice `conda` virtual environment with `./install.sh`. *[Currently this doesn't support Windows, sorry]*

If everything in the install script went OK, you can enter the created virtual environment with `source ./run_in_environment.sh`. 

#### Remaining Parts For Install

* Download the SNLI Dataset ([info](https://nlp.stanford.edu/projects/snli/)): `./scripts/get_snli_dataset.sh`
* Download the pre-trained Word2Vec model ([info](https://en.wikipedia.org/wiki/Word2vec)): `./scripts/get_word2vec.sh`

## Usage

`coming soon`


## Results

`coming soon`

## Training Details

`coming soon`

### Author

Jonathon Belotti ([thundergolfer](https://github.com/thundergolfer)) - [@jonobelotti_IO](https://twitter.com/jonobelotti_IO) - [jonathonbelotti.com](http://jonathonbelotti.com)

### License

This project is available under the [MIT License](https://choosealicense.com/licenses/mit/)
