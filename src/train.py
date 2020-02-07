from argparse import ArgumentParser

from flair.datasets import ColumnCorpus
from flair.embeddings import (FlairEmbeddings, StackedEmbeddings,
                              TransformerWordEmbeddings, WordEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import Adam

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="data directory containing txt files")
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="path to which all output during training is logged and models are saved",
    )
    parser.add_argument(
        "--word_embedding",
        type=str,
        default=None,
        help="embedding id to use with flair.WordEmbeddings, checkout "
        + "https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md "
        + "for the full list",
    )
    parser.add_argument(
        "--flair_embedding",
        type=str,
        default=None,
        help="embedding id to use with flair.FlairEmbeddings, "
        + "e.g: news-X-fast, note: 'X' will be replace with 'forward' and 'backward', checkout "
        + "https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md"
        + "for the full list",
    )
    parser.add_argument(
        "--transformer_embedding",
        type=str,
        default=None,
        help="embedding id to use with flair.TransformerWordEmbeddings, checkout "
        + "https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md "
        + "for the full list",
    )
    parser.add_argument("--use_rnn", action="store_true", help="use RNN layer, otherwise use word embeddings directly")
    parser.add_argument("--use_crf", action="store_true", help="use CRF decoder, else project directly to tag space")
    parser.add_argument("--hidden_size", type=int, default=256, help="number of hidden states in RNN")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="maximum number of epochs to train. Terminates training if this number is surpassed.",
    )
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    assert not (
        args.word_embedding is None and args.flair_embedding is None and args.transformer_embedding is None
    ), f"At least 1 embedding needs to be specified"

    columns = {0: "text", 1: "deletion"}
    corpus = ColumnCorpus(args.data_dir, columns, train_file="train.txt", dev_file="dev.txt")

    target_tag = "deletion"
    tag_dict = corpus.make_tag_dictionary(tag_type=target_tag)

    embeddings = []
    if args.word_embedding is not None:
        embeddings.append(WordEmbeddings(args.word_embedding))
    if args.flair_embedding is not None:
        for direction in ["forward", "backward"]:
            embeddings.append(FlairEmbeddings(args.flair_embedding.replace("X", direction)))
    if args.transformer_embedding is not None:
        embeddings.append(TransformerWordEmbeddings(args.transformer_embedding))

    embeddings = StackedEmbeddings(embeddings=embeddings)
    tagger = SequenceTagger(
        hidden_size=args.hidden_size,
        embeddings=embeddings,
        tag_dictionary=tag_dict,
        tag_type=target_tag,
        use_crf=args.use_crf,
        use_rnn=args.use_rnn,
    )

    trainer = ModelTrainer(tagger, corpus, optimizer=Adam)
    trainer.train(args.train_dir, learning_rate=args.lr, max_epochs=args.max_epochs, mini_batch_size=args.batch_size)
