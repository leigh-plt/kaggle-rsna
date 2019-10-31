# RSNA Intracranial Hemorrhage Detection

This is the project for [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) hosted on Kaggle in 2019.

## Train models

~~~
$ sh ./script/runner.sh -m resnet34
$ sh ./script/runner.sh -m resnet50
$ sh ./script/runner.sh -m inception_v3
$ sh ./script/runner.sh -m densenet169
~~~

## Prestage (optinal)

Model trained with converted dicom files to jpg with python script `src/images.py`, without converting change in `./script/runner.sh` flag train on image to `False`

## Submit

Final submit mean value of 4 models
