# Datasets

Download the German voice dataset from <https://commonvoice.mozilla.org/en/datasets>.
Move it to some directory. As of 2021-08-12, the dataset you get is called `cv-corpus-7.0-2021-07-21-de.tar.gz`.

Then run 

```sh
# From the base directory:
./scripts/create_train_dataset.py /path/to/cv-corpus-7.0-2021-07-21/de/ --output_path=$(pwd)/data/
```

This will create a `train.tsv` as well as copy the audio files to ones that
are numbered according to its corresponding line in `train.tsv`.
