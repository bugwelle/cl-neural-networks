# cl-neural-networks

Introduction to Neural Networks and Sequence-To-Sequence Learning at the Heidelberg University

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
