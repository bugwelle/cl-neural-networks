#!/usr/bin/env python3

# Example usage:
#  ./create_train_dataset.py /opt/cv-corpus-7.0-2021-07-21/de/ --output_path=/home/andre/Projects/uni/cl-neural-networks/data/
#

import argparse
import pandas as pd
import os
import shutil

DATASET_INPUT  = "train.tsv"
DATASET_OUTPUT = "train.tsv"

def main():
    ap = argparse.ArgumentParser("Create training dataset")

    ap.add_argument("path",          type=str, help="path to TSV files and audio files")
    ap.add_argument("--output_path", type=str, help="path for saving output files")

    args = ap.parse_args()

    if args.output_path == None or args.output_path == "":
        raise RuntimeError("Missing output path")


    input_tsv_file  = os.path.join(args.path, DATASET_INPUT)
    output_tsv_file = os.path.join(args.output_path, DATASET_OUTPUT)

    # Read TSV file into DataFrame df
    df = pd.read_csv(input_tsv_file, sep="\t")
    df = df[(df.down_votes < 2) & (df.accent.isnull()) & (df.locale == "de") & (df.sentence.str.len() < 40) & (df.gender == 'male')]
    df = df.sort_values(by="up_votes", ascending=False).head(1000)
    
    df = df.drop(columns=['accent', 'age', 'locale', 'segment', 'client_id', 'gender', 'up_votes', 'down_votes']).reset_index(drop=True)

    print("A quick glance at the data frame:")
    print(df)

    print(f"Now creating TSV file at: {output_tsv_file}")
    df.to_csv(output_tsv_file, columns=["sentence"], index=False, header=False, sep="\t")

    print(f"Now creating audio file at: {args.output_path}/audio")

    os.mkdir(os.path.join(args.output_path, 'audio'))

    for index, row in df.iterrows():
        src = os.path.join(args.path, 'clips', row["path"])
        dst = os.path.join(args.output_path, 'audio', f"{index}.mp3")

        if not os.path.exists(src):
            raise RuntimeError(f"File does not exist: {src}")
        
        # print(f"Copying {src} to {dst}")
        shutil.copyfile(src, dst)

    print("Finished. You can use 'tar -czvf audio.tar.gz ./audio' to create an archive of all files.")

if __name__ == "__main__":
    main()
