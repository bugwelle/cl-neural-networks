#!/usr/bin/env python3

# Example usage:
#  ./create_train_dataset.py /opt/cv-corpus-7.0-2021-07-21/de/ --output_path=/home/andre/Projects/uni/cl-neural-networks/data/
#

import argparse
import pandas as pd
import os
import shutil

def create_dataset(input_path, input_tsv_name, output_path, output_tsv_name, output_folder_name, max_files = 1000):

    input_tsv_file  = os.path.join(input_path, input_tsv_name)
    output_tsv_file = os.path.join(output_path, output_tsv_name)

    # Read TSV file into DataFrame df
    df = pd.read_csv(input_tsv_file, sep="\t")
    #& (df.path.str.contains('common_voice_de_1\d{7}'))
    df = df[(df.down_votes < 3) & (df.accent.isnull()) & (df.locale == "de") & (df.sentence.str.len() < 75) & (df.gender == 'male') & (~df.sentence.str.contains("[^\x00-\x7FäÄöÖüÜß]", na=False))]
    df["sentence"] = df["sentence"].str.lower()

    df = df.sort_values(by="up_votes", ascending=False).head(max_files)
    
    df = df.drop(columns=['accent', 'age', 'locale', 'segment', 'client_id', 'gender', 'up_votes', 'down_votes']).reset_index(drop=True)

    print("A quick glance at the data frame:")
    print(df)

    print(f"Now creating TSV file at: {output_tsv_file}")
    df.to_csv(output_tsv_file, columns=["sentence"], index=False, header=False, sep="\t")

    print(f"Now creating audio file at: {output_path}/{output_folder_name}")

    os.mkdir(os.path.join(output_path, output_folder_name))

    for index, row in df.iterrows():
        src = os.path.join(input_path, 'clips', row["path"])
        dst = os.path.join(output_path, output_folder_name, f"{index}.mp3")

        if not os.path.exists(src):
            raise RuntimeError(f"File does not exist: {src}")
        
        # print(f"Copying {src} to {dst}")
        shutil.copyfile(src, dst)

    print(f"Finished. You can use 'tar -czvf {output_folder_name}.tar.gz ./{output_folder_name}' to create an archive of all files.")

def main():

    ap = argparse.ArgumentParser("Create training dataset")

    ap.add_argument("path",          type=str, help="path to TSV files and audio files")
    ap.add_argument("--output_path", type=str, help="path for saving output files")

    args = ap.parse_args()

    if args.output_path == None or args.output_path == "":
        raise RuntimeError("Missing output path")

    create_dataset(args.path, "train.tsv", args.output_path, "train.tsv", "train_audio", 10000)
    create_dataset(args.path, "test.tsv", args.output_path, "test.tsv", "test_audio", 100)

if __name__ == "__main__":
    main()
