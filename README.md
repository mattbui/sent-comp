# ICCCI - Sentence Compression with deletion

## Dataset

The dataset is available at: [https://github.com/google-research-datasets/sentence-compression](https://github.com/google-research-datasets/sentence-compression). Download and store the `*.gz` files in `data/` directory.

## Requirements

This project requires python3.6+ and pytorch1.1+. It used the models and embeddings from [FLAIR framework](https://github.com/flairNLP/flairhttps://github.com/flairNLP/flair):

```bash
pip install flair
```

## Preprocess data

In order to train a sequence tagging model, the original data need to be align into sequence tagging format. To align the downloaded data:

```bash
export PRJ_HOME=<path/to/this/project>
bash $PRJ_HOME/runs/preprocess.sh
```

## Training
