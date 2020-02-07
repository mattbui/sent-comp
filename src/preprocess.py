import argparse
import glob
import gzip
import json
import logging
import os
import re
import sys
from collections import OrderedDict
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

O_tag = "O"
delete_tag = "D"
_letter_num = r"a-zA-Z0-9"
_whitespace = r"\s+"


def load_samples_from_jsonfiles(filepaths):
    samples = []
    for filepath in tqdm(filepaths, desc="Load json data"):
        with gzip.open(filepath, mode="rt", encoding="utf-8") as f:
            all_text = f.read()
        all_text = "[" + all_text + "]"
        all_text = all_text.replace("}\n\n{", "},\n{")

        samples.extend(json.loads(all_text))
    return samples


def _clean_text(text):
    text = text.lower()
    # Special case: won't, wouldn't, etc will be split into (wo, n't), (would, n't), etc
    text = re.sub(f"([a-z]+)(n't)", r"\g<1> \g<2>", text)

    text = re.sub(f"[^{_letter_num}]", " ", text)  # remove special characters
    text = re.sub(f"{_whitespace}", " ", text)  # collapse whitespaces
    text = text.strip()
    return text


def _match_word(word, sentence):
    cleaned_word = _clean_text(word)
    cleaned_words = _clean_text(word).split(" ")
    cleaned_sentence = _clean_text(sentence)
    if cleaned_words == cleaned_sentence.split(" ", len(cleaned_words))[: len(cleaned_words)]:
        return True, cleaned_sentence[len(cleaned_word) :].strip()
    return False, cleaned_sentence


def convert_json_sample_to_tagging_sample(sample):
    word_sequence = []
    label_sequence = []

    id2word_stree = {}
    id2word_graph = {}
    id2node = {}
    id2node_stree = {}
    mid2node = {}
    stem2words = {}
    stem2nodes = {}

    for node in sample["source_tree"]["node"]:
        for word in node["word"]:
            id2word_stree[word["id"]] = word
            id2node_stree[word["id"]] = node

    for node in sample["graph"]["node"]:
        if "mid" in node:
            mid = node["mid"]
            if mid not in mid2node:
                mid2node[mid] = []
            mid2node[mid].append(node)

        for word in node["word"]:
            id2word_graph[word["id"]] = word
            id2node[word["id"]] = node
            stem = word["stem"]
            if stem not in stem2words:
                stem2words[stem] = []
                stem2nodes[stem] = []
            stem2words[stem].append(word)
            stem2nodes[stem].append(node)

    remain_compression_text = _clean_text(sample["compression"]["text"])
    for edge in sample["compression_untransformed"]["edge"]:
        word = id2word_stree.get(edge["child_id"], None)
        if word is None:
            continue

        stree_node = id2node_stree[word["id"]]
        logging.debug(f"{word['form']} | {stree_node['form']} | {remain_compression_text[:10]}")
        matched, remain_compression_text = _match_word(stree_node["form"], remain_compression_text)
        if matched:
            for word in stree_node["word"]:
                id2word_stree[word["id"]]["deletion"] = O_tag
            continue

        matched, remain_compression_text = _match_word(word["form"], remain_compression_text)
        if matched:
            word["deletion"] = O_tag
            continue

        graph_word = id2word_graph.get(word["id"], None)
        if graph_word is None:
            continue

        mid = id2node[word["id"]].get("mid", None)
        if mid is not None:
            matched = False
            for node in mid2node[mid]:
                for word in node["word"]:
                    logging.debug(f"candidate word in node via mid: {word['form']}")
                    matched_, remain_compression_text = _match_word(word["form"], remain_compression_text)
                    if matched_:
                        id2word_stree[word["id"]]["deletion"] = O_tag
                    matched = matched_ or matched
                if matched:
                    break
            if matched:
                continue

        stem = graph_word["stem"]
        matched = False
        for node in stem2nodes[stem]:
            for word in node["word"]:
                logging.debug(f"candidate word in node via stem: {word['form']}")
                matched_, remain_compression_text = _match_word(word["form"], remain_compression_text)
                if matched_:
                    id2word_stree[word["id"]]["deletion"] = O_tag
                    matched = True
            if matched:
                break
        if matched:
            continue
        for word in stem2words[stem]:
            logging.debug(f"candidate word via stem: {node['form']}")
            matched, remain_compression_text = _match_word(word["form"], remain_compression_text)
            if matched:
                id2word_stree[word["id"]]["deletion"] = O_tag
                break

    id2word = OrderedDict(sorted(id2word_stree.items()))
    for word in id2word.values():
        if word["id"] == -1:
            continue
        word_sequence.append(word["form"])
        label_sequence.append(word.get("deletion", delete_tag))

    return word_sequence, label_sequence, remain_compression_text


def convert_json_samples_to_tagging_samples(samples, thread=cpu_count()):
    with Pool(thread) as p:
        tagging_samples = list(
            tqdm(
                p.imap(convert_json_sample_to_tagging_sample, samples, chunksize=32),
                total=len(samples),
                desc="Convert json samples to tagging samples",
            )
        )

    num_error = sum([len(remain_text) > 0 for _, _, remain_text in tagging_samples])
    logging.info(f"{num_error}/{len(samples)} json samples are unable to align into tagging samples")
    return [
        (word_sequence, label_sequence)
        for word_sequence, label_sequence, remain_text in tagging_samples
        if len(remain_text) == 0
    ]


def write_tagging_samples_to_file(samples, filepath):
    with open(filepath, "w", encoding="utf-8") as outfile:
        outfile.write("-DOCSTART- O\n\n")
        for word_sequence, label_sequence in samples:
            for word, label in zip(word_sequence, label_sequence):
                outfile.write(f"{word} {label}\n")
            outfile.write("\n")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="data directory containing gz files")
    parser.add_argument("--thread", type=int, default=cpu_count(), help="# threads used for aligning data")

    args = parser.parse_args()

    thread = min(args.thread, cpu_count())
    train_filepaths = glob.glob(f"{args.data_dir}/*train*")
    dev_filepath = os.path.join(args.data_dir, "comp-data.eval.json.gz")

    logging.info("Process train data")
    train_samples = load_samples_from_jsonfiles(train_filepaths)
    train_tagging_samples = convert_json_samples_to_tagging_samples(train_samples, thread=thread)
    write_tagging_samples_to_file(train_tagging_samples, os.path.join(args.data_dir, "train.txt"))
    del train_samples, train_tagging_samples

    logging.info("Process dev data")
    dev_samples = load_samples_from_jsonfiles([dev_filepath])
    dev_tagging_samples = convert_json_samples_to_tagging_samples(dev_samples, thread=thread)
    write_tagging_samples_to_file(dev_tagging_samples, os.path.join(args.data_dir, "dev.txt"))
