# Speec2Text with JoeyNMT

This repository is our project for the seminar "Introduction to Neural Networks and Sequence-To-Sequence Learning" at the Heidelberg University.
Our project is speech to text transformations using [JoeyNMT](https://github.com/joeynmt/joeynmt.git).

## Table of Contents

- Run our approach in [colab](./colab)
- See our [configurations](./config) for each model
- Our [data](./data) also includes [results](./data/results) of our models
- See [report](./report) for more details
- [Scripts](./scripts) to process our data
- [Speech2Text](./speech2text)

## Setup

```sh
git clone git@github.com:bugwelle/cl-neural-networks.git
cd cl-neural-networks

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies for our scripts
pip3 install -r requirements.txt

# Install JoeyNMT dependencies
cd speech2text
pip3 install -r requirements.txt
```

To create the dataset, see `data/README.md`.


## Notes on this project

The folder `speech2text` is originally the [JoeyNMT project](https://github.com/joeynmt/joeynmt.git).
To work with it, we cloned it (commit [`c53f64e1910fb65f51a078dbd39d78d2fa16e26a`](https://github.com/joeynmt/joeynmt/tree/c53f64e1910fb65f51a078dbd39d78d2fa16e26a)) and modified it so that it works with audio files.

This is how we set up the folder:

```sh
git clone https://github.com/joeynmt/joeynmt.git speech2text
# Remove git folder and docs
rm -rf speech2text/.git speech2text/.github speech2text/docs
```

If you want to see what we've changed to JoeyNMT, have a look at this diff:
<https://github.com/bugwelle/cl-neural-networks/compare/ee3b71883b100ced0118366dac46f57f804773dc...main>


## How to train and test your model

First, install JoeyNMT:

```sh
cd speech2text/
pip3 install .
cd ..
```

Then train your model (note, you need actual audio files,
see previous sections):

```sh
python3 -m joeynmt train config/speech.yaml
```

After training, you can use your trained model to translate
audio files.

This works by creating a text file that contains a single line
with the path to your audio file (can also be a relative path).
For example:

```txt
data/test_audio/18.mp3
```

You can then pipe this file into JoeyNMT using:

```sh
python3 -m joeynmt translate config/speech.yaml < text.txt
```

This will give you the translated text.
