# High-order Joint Constituency and Dependency Parsing

## Setup

The following packages should be installed:
* [`PyTorch`](https://github.com/pytorch/pytorch): >= 1.12.1
* [`Transformers`](https://github.com/huggingface/transformers): >= 4.2

Please modify `data/model-conf/data.ini` according to paths of your own dataset.

You can download the processed [PTB-train](https://drive.google.com/file/d/1rBOXMrfWlpBEPY7zZk_24ENh-qZNVTSN/view?usp=sharing), where incompatible instances are removed. The meaning of compatibility can be viewd in our paper. 

## Run

Try the following commands to train second-order Joint2o model:
```sh
python src/parse.py -c data/model-conf/bert-ptb/joint2o-bi.ini -d 0
```

To make predictions:
```sh
python src/pred.py -d 0 -c data/model-conf/bert-ptb/joint2o-bi.ini \
    --path exp/bert-e25-ptb/joint2o-bi-s1/model \
    --data_con data/ptb/test.pid \
    --data_dep data/ptb/test.conllx \
    --pred exp/joint2o-bi-s1/test.pred \
    --decode satta
```

## Contact

If you have any questions, feel free to contact me via [emails](mailto:yanggangu@outlook.com).


## Citation

If you are interested in our work, please cite
```bib
@inproceedings{gu-etal-2024-high-order,
  title     = {High-order Joint Constituency and Dependency Parsing},
  author    = {Yanggan, Gu  and
               Yang, Hou  and
               Zhefeng, Wang  and
               Xinyu, Duan  and
               Zhenghua, Li},
  booktitle = {Proceedings of LREC-COLING},
  year      = {2024},
  address   = {Torino, Italia},
  publisher = {International Committee on Computational Linguistics}
}
```