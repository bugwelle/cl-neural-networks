# Results

We have different models that we trained.
The results vary a lot.
There is still room for improvement.

This directory contains the results of each model for the test dataset.
You can compare it to `train.tsv`.

```sh
for i in {0..99}; do
    echo "--------------";
    echo $i;
    python3 -m joeynmt translate \
        config/speech_model_c.yaml < \
        <(echo "data/test_audio/$i.mp3"); 
done |& tee log.txt
```
