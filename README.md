# RT-GENE-main
#EasyGaze Net

Chengyihua RT-Gene-main(https://github.com/yihuacheng/RT-Gene)
Self learning and baseline
==Modified the problem of using label file directly for facial feature extraction==
![image](https://user-images.githubusercontent.com/95728828/178435734-cdc175f7-7d05-4d6d-baa6-16903d5e050d.png)

## Introduction

We provide two projects for leave-one-person-out evaluation and the evaluation of common training-test split.
They have the same architecture but different `train.py` and `test.py`.

Each project contains following files/folders.

- `model.py`, the model code.
- `train.py`, the entry for training.
- `test.py`, the entry for testing.
- `config/`, this folder contains the config of the experiment in each dataset. To run our code, **you should write your own** `config.yaml`. 
- `reader/`, the code for reading data. You can use the provided reader or write your own reader.

## Getting Started

### Writing your own *config.yaml*

Normally, for training, you should change 

1. `train.save.save_path`, The model is saved in the `$save_path$/checkpoint/`.
2. `train.data.image`, This is the path of image,  please use the provided data processing code in <a href="http://phi-ai.org/GazeHub/" target="_blank">*GazeHub*</a>.
3. `train.data.label`, This is the path of label.
4. `reader`, This indicates the used reader. It is the filename in `reader` folder, e.g., *reader/reader_mpii.py* ==> `reader: reader_mpii`.

For test, you should change 

1. `test.load.load_path`, it is usually the same as `train.save.save_path`. The test result is saved in `$load_path$/evaluation/`.
2. `test.data.image`, it is usually the same as `train.data.image`.
3. `test.data.label`, it is usually the same as `train.data.label`.

### Training

In the leaveout folder, you can run

```
python train.py config/config_mpii.yaml 0
```

This means the code will run with `config_mpii.yaml` and use the `0th` person as the test set.

You also can run

```
bash run.sh train.py config/config_mpii.yaml
```

This means the code will perform leave-one-person-out training automatically.   
`run.sh` performs iteration, you can change the iteration times in `run.sh` for different datasets, e.g., set the iteration times as `4` for four-fold validation.

In the traintest folder, you can run

```
python train.py config/config_mpii.yaml
```

### Test

In the leaveout folder, you can run

```
python test.py config/config_mpii.yaml 0
```

or

```
bash run.sh test.py config/config_mpii.yaml
```

In the traintest folder, you can run

```
python test.py config/config_mpii.yaml
```

### Result

After training or test, you can find the result from the `$save_path$` in `config_mpii.yaml`. 
