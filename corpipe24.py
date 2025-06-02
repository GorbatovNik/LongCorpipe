#!/usr/bin/env python3

# This file is part of CorPipe <https://github.com/ufal/crac2024-corpipe>.
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import functools
import json
import math
import os
import pymorphy2
import re
import shutil
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Tuple

from clusterer import merge_clusters

morph = pymorphy2.MorphAnalyzer()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import transformers

import udapi
import udapi.block.corefud.movehead
import udapi.block.corefud.removemisc
from udapi.block.read.conllu import Conllu

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
parser.add_argument("--dev", default=None, nargs="*", type=str, help="Predict dev (treebanks).")
parser.add_argument("--exp", default="", type=str, help="Exp name.")
parser.add_argument("--load", default=[], type=str, nargs="*", help="Models to load.")
parser.add_argument("--right", default=50, type=int, help="Reserved space for right context, if any.")
parser.add_argument("--segment", default=512, type=int, help="Segment size")
parser.add_argument("--test", default=None, nargs="*", type=str, help="Predict test (treebanks).")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--treebanks", default=[], nargs="+", type=str, help="Data.")
parser.add_argument("--treebank_id", default=False, action="store_true", help="Use treebank id.")
parser.add_argument("--zeros_per_parent", default=2, type=int, help="Zeros per parent.")


class Dataset:
    TOKEN_EMPTY = "[TOKEN_EMPTY]"
    TOKEN_CLS = "[TOKEN_CLS]"
    TOKEN_TREEBANK = "[TOKEN_TREEBANK{}]"
    ZDEPREL_PAD = 0
    ZDEPREL_NONE = 1

    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizerFast, treebank_id: int) -> None:
        self._cls = tokenizer.cls_token_id
        self._sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self._path = path
        self._treebank_token = []
        if treebank_id:  # 0 is deliberately considered as no treebank id
            treebank_token = tokenizer.vocab[self.TOKEN_TREEBANK.format(treebank_id - 1)]
            if self._cls is None:
                self._cls = treebank_token
            else:
                self._treebank_token = [treebank_token]
        if self._cls is None:
            self._cls = tokenizer.vocab[self.TOKEN_CLS]

        # Create the tokenized documents if they do not exist
        docs, new_doc = [], []
        docs_flu, new_doc_flu = [], []
        for doc in udapi.block.read.conllu.Conllu(files=[path]).read_documents():
            for tree in doc.trees:
                if tree.newdoc is not None and new_doc:
                    docs.append(new_doc)
                    new_doc = []
                    docs_flu.append(new_doc_flu)
                    new_doc_flu = []
                words, coref_mentions = [], set()
                new_doc_flu.append([])
                for node in tree.descendants:
                    words.append(node.form)
                    coref_mentions.update(node.coref_mentions)
                    new_doc_flu[-1].append((node.form, node.lemma, node.upos))
                for enode in tree.empty_nodes:
                    coref_mentions.update(enode.coref_mentions)

                dense_mentions = []
                for mention in [mention for mention in coref_mentions if not mention.head.is_empty()]:
                    span = [word for word in mention.words if not word.is_empty()]
                    start = end = span.index(mention.head)
                    while start > 0 and span[start - 1].ord + 1 == span[start].ord: start -= 1
                    while end < len(span) - 1 and span[end].ord + 1 == span[end + 1].ord: end += 1
                    dense_mentions.append(((span[start].ord - 1, span[end].ord - 1), mention.entity.eid,
                                           start > 0 or end + 1 < len(span)))
                dense_mentions = sorted(dense_mentions, key=lambda x: (x[0][0], -x[0][1], x[2]))

                mentions = []
                for i, mention in enumerate(dense_mentions):
                    if i and dense_mentions[i - 1][0] == mention[0]:
                        print(
                            f"Multiple same mentions {mention[2]}/{dense_mentions[i - 1][2]} in sent_id {tree.sent_id}: {tree.get_sentence()}",
                            flush=True)
                        continue
                    mentions.append((mention[0][0], mention[0][1], mention[1]))

                zero_mentions = []
                for mention in [mention for mention in coref_mentions if mention.head.is_empty()]:
                    if len(mention.words) > 1:
                        print(
                            f"A empty-node-head mention with multiple words {mention.words} in sent_id {tree.sent_id}: {tree.get_sentence()}",
                            flush=True)
                    assert len(mention.head.deps) >= 1
                    zero_mentions.append(
                        (mention.head.deps[0]["parent"].ord - 1, mention.head.deps[0]["deprel"], mention.entity.eid))
                zero_mentions = sorted(zero_mentions)
                new_doc.append((words, mentions, zero_mentions))
            if new_doc:
                docs.append(new_doc)
            if new_doc_flu:
                docs_flu.append(new_doc_flu)

        self.docs_flu = docs_flu
        # Tokenize the data, generate stack operations and subword mentions
        self.docs = []
        for doc in docs:
            new_doc = []
            for words, mentions, zero_mentions in doc:
                subwords, word_indices, word_tags, subword_mentions, stack = [], [], [], [], []
                for i in range(len(words)):
                    word_indices.append(len(subwords))
                    word = (" " if "robeczech" in tokenizer.name_or_path else "") + words[i]
                    subword = tokenizer.encode(word, add_special_tokens=False)
                    assert len(subword) > 0
                    if subword[
                        0] == 6 and "xlm-r" in tokenizer.name_or_path:  # Hack: remove the space-only token in XLM-R
                        subword = subword[1:]
                    assert len(subword) > 0
                    subwords.extend(subword)

                    tag = [str(len(stack))]
                    for _ in range(2):
                        for j in reversed(range(len(stack))):
                            start, end, eid = stack[j]
                            if end == i:
                                tag.append(f"POP:{len(stack) - j}")
                                subword_mentions.append((start, word_indices[-1], eid))
                                stack.pop(j)
                        while mentions and mentions[0][0] == i:
                            tag.append("PUSH")
                            stack.append((word_indices[-1], mentions[0][1], mentions[0][2]))
                            mentions = mentions[1:]
                    word_tags.append(",".join(tag))
                assert len(stack) == 0

                word_zdeprels = [[] for _ in range(len(words))]
                for parent, deprel, eid in zero_mentions:
                    word_zdeprels[parent].append(deprel)
                    subword_mentions.append((word_indices[parent], -len(word_zdeprels[parent]), eid))
                subword_mentions = sorted(subword_mentions, key=lambda x: (x[0], -x[1]))
                new_doc.append((subwords, word_indices, word_tags, word_zdeprels, subword_mentions))
            self.docs.append(new_doc)

        for doc in self.docs:
            for _, _, word_tags, _, _ in doc:
                for i in range(len(word_tags)):
                    word_tags[i] = ",".join(word_tags[i].split(",")[1:])

    @staticmethod
    def create_tags(trains: list[Dataset]) -> list[str]:
        tags = set()
        for train in trains:
            for doc in train.docs:
                for _, _, word_tags, _, _ in doc:
                    tags.update(word_tags)
        return sorted(tags)

    @staticmethod
    def create_zdeprels(trains: list[Dataset]) -> list[str]:
        zdeprels = set()
        for train in trains:
            for doc in train.docs:
                for _, _, _, word_zdeprels, _ in doc:
                    zdeprels.update(zdeprel for zdeprels in word_zdeprels for zdeprel in zdeprels)
        return ["[PAD]", "[NONE]"] + sorted(zdeprels)  # Respect ZDEPREL_PAD and ZDEPREL_NONE

    @staticmethod
    def allowed_tag_transitions(tags: list[str], depth: int) -> np.array:
        tags = [f"{d}{',' if tag else ''}{tag}" for d in range(depth) for tag in tags]
        allowed = np.zeros([len(tags), len(tags)], np.float32)
        for i, tag_i in enumerate(tags):
            for j, tag_j in enumerate(tags):
                i_parts = tag_i.split(",")
                i_depth = int(i_parts[0])
                j_depth = int(tag_j.split(",")[0])
                for command in i_parts[1:]:
                    i_depth += 1 if command == "PUSH" else -1
                allowed[i, j] = i_depth == j_depth
        return allowed

    def pipeline(self, tags_map: dict[str, int], zdeprels_map: dict[str, int], train: bool,
                 args: argparse.Namespace) -> tf.data.Dataset:
        def generator():
            tid = len(self._treebank_token)
            for doc in self.docs:
                p_subwords, p_subword_mentions = [], []
                for doc_i, (subwords, word_indices, word_tags, word_zdeprels, subword_mentions) in enumerate(doc):
                    subword_mentions = [(s, e, eid) for s, e, eid in subword_mentions if e >= -args.zeros_per_parent]
                    if len(subwords) + 4 + tid > args.segment:
                        print("Truncating a long sentence during prediction")
                        subwords = subwords[:args.segment - 4 - tid]
                    assert len(subwords) + 4 + tid <= args.segment
                    if len(subwords) + 4 + tid <= args.segment:
                        right_reserve = min((args.segment - 4 - tid - len(subwords)) // 2, args.right or 0)
                        context = min(args.segment - 4 - tid - len(subwords) - right_reserve, len(p_subwords))
                        word_indices = [context + 2 + tid + i for i in word_indices + [len(subwords)]]
                        e_subwords = [self._cls, *self._treebank_token, *p_subwords[-context:], self._sep, *subwords,
                                      self._sep]
                        if args.right is not None:
                            i = doc_i + 1
                            while i < len(doc) and len(e_subwords) + 1 < args.segment:
                                e_subwords.extend(doc[i][0][:args.segment - len(e_subwords) - 1])
                                i += 1
                        e_subwords.append(self._sep)

                        output = (e_subwords, word_indices)

                        yield output

                    p_subword_mentions.extend(
                        (s + len(p_subwords), e if e < 0 else e + len(p_subwords), eid) for s, e, eid in
                        subword_mentions)
                    p_subwords.extend(subwords)

        output_signature = (tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32))

        pipeline = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        pipeline = pipeline.cache()
        pipeline = pipeline.apply(tf.data.experimental.assert_cardinality(sum(1 for _ in pipeline)))
        return pipeline

    def save_mentions(self, path: str, mentions: list[
        list[tuple[int, int, int]]]):  # , zero_mentions: list[list[tuple[int, str, int]]]) -> None:
        doc = udapi.block.read.conllu.Conllu(files=[self._path]).read_documents()[0]
        udapi.block.corefud.removemisc.RemoveMisc(attrnames="Entity,SplitAnte,Bridge").apply_on_document(doc)
        entities = {}
        for i, tree in enumerate(doc.trees):
            tree.empty_nodes = []  # Drop existing empty nodes
            for node in tree.descendants:  # Remove references to empty nodes also from DEPS, by replacing them by the main dependency edge
                if "." in node.raw_deps:
                    node.raw_deps = f"{node.parent.ord}:{node.deprel}"
            ords = {}
            for mention in mentions[i]:
                if not isinstance(mention, ZeroMention):
                    continue
                parent, deprel, eid = mention.begin, mention.zdeprel, mention.cluster
                tree.create_empty_child()
                ords[parent] = ords.get(parent, 0) + 1
                tree.empty_nodes[-1].ord = f"{parent + 1}.{ords[parent]}"
                tree.empty_nodes[-1].raw_deps = f"{parent + 1}:{deprel}"
                if not eid in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention([tree.empty_nodes[-1]], entity=entities[eid])
            nodes = tree.descendants_and_empty
            for mention in mentions[i]:
                if isinstance(mention, ZeroMention):
                    continue
                start, end, eid = mention.begin, mention.end, mention.cluster
                if not eid in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention([node for node in nodes if start <= node.ord - 1 <= end],
                                              entity=entities[eid])
        doc._eid_to_entity = {entity._eid: entity for entity in sorted(entities.values())}
        udapi.block.corefud.movehead.MoveHead(bugs='ignore').apply_on_document(doc)
        udapi.block.write.conllu.Conllu(files=[path]).apply_on_document(doc)

    def save_mentions23(self, path: str, mentions: list[list[tuple[int, int, int]]]) -> None:
        doc = udapi.block.read.conllu.Conllu(files=[self._path]).read_documents()[0]
        udapi.block.corefud.removemisc.RemoveMisc(attrnames="Entity,SplitAnte,Bridge").apply_on_document(doc)

        entities = {}
        for i, tree in enumerate(doc.trees):
            nodes = tree.descendants_and_empty
            for start, end, eid in mentions[i]:
                if not eid in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention(nodes[start:end + 1], entity=entities[eid])
        doc._eid_to_entity = {entity._eid: entity for entity in sorted(entities.values())}
        udapi.block.corefud.movehead.MoveHead(bugs='ignore').apply_on_document(doc)
        udapi.block.write.conllu.Conllu(files=[path]).apply_on_document(doc)


@dataclass
class Mention:
    sent_id: int
    begin: int  # first token index in sentence
    end: int  # last token index in sentence (inclusive)
    info: list[Tuple[str, str, str]]
    cluster: int | None = None


class ZeroMention(Mention):
    zdeprel: str


class Model(tf.keras.Model):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, tags: list[str], zdeprels: list[str],
                 args: argparse.Namespace) -> None:
        super().__init__()
        self._tags = tags
        self._zdeprels = zdeprels
        self._args = args
        self._tokenizer = tokenizer

        assert tags[0] == ""  # Used as a boundary tag in CRF
        self._allowed_tag_transitions = tf.constant(Dataset.allowed_tag_transitions(tags, args.depth + 1))
        self._boundary_logits = tf.cast(tf.range(self._allowed_tag_transitions.shape[0]) > 0, tf.float32) * -1e6

        self._encoder = transformers.TFMT5EncoderModel if "mt5" in args.encoder.lower() else transformers.TFAutoModel
        if not args.load:
            self._encoder = self._encoder.from_pretrained(
                args.encoder, from_pt=any(m in args.encoder.lower() for m in
                                          ["rubert", "herbert", "flaubert", "litlat", "roberta-base-ca", "spanbert",
                                           "xlm-v", "infoxlm"]))
        else:
            self._encoder = self._encoder.from_config(transformers.AutoConfig.from_pretrained(args.encoder))
            self._encoder(tf.constant([[0]], dtype=tf.int32), attention_mask=tf.constant([[1.]], dtype=tf.float32))
        self._encoder.resize_token_embeddings(len(tokenizer.vocab))
        self._dense_hidden_q = tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu,
                                                     name="dense_hidden_q")
        self._dense_hidden_k = tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu,
                                                     name="dense_hidden_k")
        self._dense_hidden_tags = tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu,
                                                        name="dense_hidden_tags")
        self._dense_q = tf.keras.layers.Dense(self._encoder.config.hidden_size, use_bias=False, name="dens_q")
        self._dense_k = tf.keras.layers.Dense(self._encoder.config.hidden_size, use_bias=False, name="dens_k")
        self._dense_tags = tf.keras.layers.Dense(len(tags), name="dense_tags")
        self._dense_hidden_zeros = [
            tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu,
                                  name=f"dense_hidden_zeros_{i}") for i in range(args.zeros_per_parent)]
        self._dense_zeros = [tf.keras.layers.Dense(self._encoder.config.hidden_size, name=f"dense_zeros_{i}") for i in
                             range(args.zeros_per_parent)]
        self._dense_hidden_zdeprels = [
            tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu,
                                  name=f"dense_hidden_zdeprels_{i}") for i in range(args.zeros_per_parent)]
        self._dense_zdeprels = [tf.keras.layers.Dense(len(zdeprels), name=f"dense_zdeprels_{i}") for i in
                                range(args.zeros_per_parent)]

        if args.load:
            self.compute_tags(tf.ragged.constant([[0]]), tf.ragged.constant([[0]]))
            self.compute_antecedents(tf.zeros([1, 1, self._encoder.config.hidden_size]),
                                     tf.zeros([1, args.zeros_per_parent, self._encoder.config.hidden_size]),
                                     *[tf.ragged.constant([[[0, 0]]], dtype=tf.int32, ragged_rank=1,
                                                          inner_shape=(2,))] * 2)
            self.built = True
            self.load_weights(args.load[0])

    def crf_decode(self, logits: tf.RaggedTensor, crf_weights: tf.Tensor) -> tf.RaggedTensor:
        boundary_logits = tf.broadcast_to(self._boundary_logits,
                                          [logits.bounding_shape(0), 1, len(self._boundary_logits)])
        logits = tf.concat([boundary_logits, logits, boundary_logits], axis=1)
        predictions, _ = tfa.text.crf_decode(logits.to_tensor(), crf_weights, logits.row_lengths())
        predictions = tf.RaggedTensor.from_tensor(predictions, logits.row_lengths())
        predictions = predictions[:, 1:-1]
        return predictions

    @tf.function(experimental_relax_shapes=True)
    def compute_tags(self, subwords, word_indices, training=False) -> tuple[
        tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
        if training or subwords.bounding_shape(0) > 0:
            embeddings = self._encoder(subwords.to_tensor(),
                                       attention_mask=tf.sequence_mask(subwords.row_lengths(), dtype=tf.float32),
                                       training=training).last_hidden_state
        else:
            # During prediction, we need to correctly handle batches of size 0 when using multiple GPUs
            embeddings = tf.zeros([0, 0, self._encoder.config.hidden_size])
        words = tf.gather(embeddings, word_indices[:, :-1], batch_dims=1)
        tag_logits = self._dense_tags(self._dense_hidden_tags(words))

        zero_embeddings = []
        for i in range(self._args.zeros_per_parent):
            zero_embeddings.append(
                self._dense_zeros[i](self._dense_hidden_zeros[i](tf.concat([embeddings] + zero_embeddings, axis=-1))))
        zdeprel_logits = []
        for i in range(self._args.zeros_per_parent):
            zdeprel_logits.append(self._dense_zdeprels[i](
                self._dense_hidden_zdeprels[i](tf.gather(zero_embeddings[i], word_indices[:, :-1], batch_dims=1))))
        return embeddings, tf.concat(zero_embeddings, axis=1), tag_logits, tf.stack(zdeprel_logits, axis=-2)

    @tf.function(experimental_relax_shapes=True)
    def compute_antecedents(self, embeddings, zero_embeddings, previous, mentions) -> tf.RaggedTensor:
        mentions_embedded = tf.gather(embeddings, tf.math.maximum(mentions, 0), batch_dims=1).values
        mentions_embedded = tf.reshape(mentions_embedded, [-1, np.prod(mentions_embedded.shape[-2:])])
        zero_mentions_embedded = tf.gather(zero_embeddings,
                                           self._args.zeros_per_parent * mentions[..., 0] + tf.math.maximum(
                                               -mentions[..., 1] - 1, 0), batch_dims=1).values
        zero_mentions_embedded = tf.tile(zero_mentions_embedded, [1, 2])
        mentions_embedded = tf.where(mentions[..., 1:].values >= 0, mentions_embedded, zero_mentions_embedded)
        queries = mentions.with_values(self._dense_q(self._dense_hidden_q(mentions_embedded)))
        keys_mentions = mentions.with_values(self._dense_k(self._dense_hidden_k(mentions_embedded)))

        previous_embedded = tf.gather(embeddings, tf.math.maximum(previous, 0), batch_dims=1).values
        previous_embedded = tf.reshape(previous_embedded, [-1, mentions_embedded.shape[-1]])
        zero_previous_embedded = tf.gather(zero_embeddings,
                                           self._args.zeros_per_parent * previous[..., 0] + tf.math.maximum(
                                               -previous[..., 1] - 1, 0), batch_dims=1).values
        zero_previous_embedded = tf.tile(zero_previous_embedded, [1, 2])
        previous_embedded = tf.where(previous[..., 1:].values >= 0, previous_embedded, zero_previous_embedded)
        keys_previous = previous.with_values(self._dense_k(self._dense_hidden_k(previous_embedded)))
        keys = tf.concat([keys_previous, keys_mentions], axis=1)
        weights = tf.matmul(queries.to_tensor(), keys.to_tensor(), transpose_b=True) / (self._dense_q.units ** 0.5)
        return weights

    # def predict_best_antecedents(self, dataset: Dataset, pipeline: tf.data.Dataset) -> tuple[list[list[tuple[int, int, int]]], list[list[tuple[int, str, int]]]]:
    #     tid = len(dataset._treebank_token)

    #     results, results_zeros, entities = [], [], 0
    #     doc_mentions, doc_subwords, sent_id = [], 0, 0
    #     for b_subwords, b_word_indices in pipeline:
    #         b_embeddings, b_zero_embeddings, b_tag_logits, b_zdeprel_logits = self.compute_tags(b_subwords, b_word_indices)

    #         b_size = b_word_indices.shape[0]
    #         b_tag_logits = b_tag_logits.with_values(tf.math.log_softmax(tf.tile(b_tag_logits.values, [1, self._args.depth + 1]), axis=-1))
    #         b_tags = self.crf_decode(b_tag_logits, (1 - self._allowed_tag_transitions) * -1e6)
    #         b_zdeprels = b_zdeprel_logits.with_values(tf.argmax(b_zdeprel_logits.values, axis=-1))

    #         b_previous, b_mentions, b_refs = [], [], []
    #         for b in range(b_size):
    #             word_indices, tags, zdeprels = b_word_indices[b].numpy(), b_tags[b].numpy(), b_zdeprels[b].numpy()
    #             if word_indices[0] == 2 + tid:
    #                 doc_mentions, doc_subwords, sent_id = [], 0, 0

    #             # Decode mentions
    #             mentions, stack = [], []
    #             for i, tag in enumerate(self._tags[tag % len(self._tags)] for tag in tags):
    #                 for command in tag.split(","):
    #                     if command == "PUSH":
    #                         stack.append(i)
    #                     elif command.startswith("POP:"):
    #                         j = int(command.removeprefix("POP:"))
    #                         if len(stack):
    #                             j = len(stack) - (j if j <= len(stack) else 1)
    #                             mentions.append((stack.pop(j), i, None))
    #                     elif command:
    #                         raise ValueError(f"Unknown command '{command}'")
    #             while len(stack):
    #                 mentions.append((stack.pop(), len(tags) - 1, None))

    #             # Decode zero mentions
    #             for i, zdeprel in enumerate(zdeprels):
    #                 for j in range(self._args.zeros_per_parent):
    #                     if zdeprel[j] == Dataset.ZDEPREL_PAD or zdeprel[j] == Dataset.ZDEPREL_NONE:
    #                         break
    #                     mentions.append((i, -j - 1, self._zdeprels[zdeprel[j]]))

    #             # Prepare inputs for antecedent prediction
    #             mentions = sorted(set(mentions), key=lambda x: (x[0], -x[1]))
    #             offset = doc_subwords - (word_indices[0] - 2 - tid)
    #             results.append([]), results_zeros.append([]), b_previous.append([]), b_mentions.append([]), b_refs.append([])
    #             for doc_mention in doc_mentions:
    #                 if doc_mention[0] < offset: continue
    #                 b_previous[-1].append([doc_mention[0] - offset + 1 + tid, doc_mention[1] if doc_mention[1] < 0 else doc_mention[1] - offset + 1 + tid])
    #                 b_refs[-1].append(doc_mention[2])
    #             for mention in mentions:
    #                 if mention[2] is not None:
    #                     result_mention = [mention[0], mention[2], None, sent_id]
    #                     # results_zeros[-1].append(result_mention)
    #                 else:
    #                     result_mention = [mention[0], mention[1], None, sent_id]
    #                 results[-1].append(result_mention)
    #                 b_refs[-1].append(result_mention)
    #                 b_mentions[-1].append([word_indices[mention[0]], mention[1] if mention[1] < 0 else word_indices[mention[1]]])
    #                 doc_mentions.append([doc_subwords + word_indices[mention[0]] - word_indices[0],
    #                                      mention[1] if mention[1] < 0 else doc_subwords + word_indices[mention[1]] - word_indices[0], result_mention])
    #             doc_subwords += word_indices[-1] - word_indices[0]
    #             sent_id += 1

    #         # Decode antecedents
    #         if sum(len(mentions) for mentions in b_mentions) == 0: continue
    #         b_antecedents = self.compute_antecedents(
    #             b_embeddings, b_zero_embeddings, tf.ragged.constant(b_previous, dtype=tf.int32, ragged_rank=1, inner_shape=(2,)),
    #             tf.ragged.constant(b_mentions, dtype=tf.int32, ragged_rank=1, inner_shape=(2,)))
    #         for b in range(b_size):
    #             len_prev, mentions, refs, antecedents = len(b_previous[b]), b_mentions[b], b_refs[b], b_antecedents[b].numpy()
    #             for i in range(len(mentions)):
    #                 j = i - 1
    #                 while j >= 0 and mentions[j][0] == mentions[i][0]:
    #                     antecedents[i, j + len_prev] = antecedents[i, i + len_prev] - 1
    #                     j -= 1
    #                 j = np.argmax(antecedents[i, :i + len_prev + 1])
    #                 refs[i + len_prev][2] = refs[j]

    #     return results #, results_zeros

    def predict(self, dataset: Dataset, pipeline: tf.data.Dataset) -> tuple[
        list[list[tuple[int, int, int]]], list[list[tuple[int, str, int]]]]:
        tid = len(dataset._treebank_token)

        results, results_zeros, entities = [], [], 0
        doc_mentions, doc_subwords = [], 0
        doc_num = -1
        sent_id = 0
        for b_subwords, b_word_indices in pipeline:
            b_embeddings, b_zero_embeddings, b_tag_logits, b_zdeprel_logits = self.compute_tags(b_subwords,
                                                                                                b_word_indices)
            b_size = b_word_indices.shape[0]
            b_tag_logits = b_tag_logits.with_values(
                tf.math.log_softmax(tf.tile(b_tag_logits.values, [1, self._args.depth + 1]), axis=-1))
            b_tags = self.crf_decode(b_tag_logits, (1 - self._allowed_tag_transitions) * -1e6)
            b_zdeprels = b_zdeprel_logits.with_values(tf.argmax(b_zdeprel_logits.values, axis=-1))

            b_previous, b_mentions, b_refs = [], [], []
            for b in range(b_size):
                word_indices, tags, zdeprels = b_word_indices[b].numpy(), b_tags[b].numpy(), b_zdeprels[b].numpy()
                if word_indices[0] == 2 + tid:
                    doc_mentions, doc_subwords = [], 0
                    doc_num += 1
                    print(f"doc number {doc_num}")
                    results.append([])
                    sent_id = 0

                # Decode mentions
                mentions, stack = [], []
                for i, tag in enumerate(self._tags[tag % len(self._tags)] for tag in tags):
                    for command in tag.split(","):
                        if command == "PUSH":
                            stack.append(i)
                        elif command.startswith("POP:"):
                            j = int(command.removeprefix("POP:"))
                            if len(stack):
                                j = len(stack) - (j if j <= len(stack) else 1)
                                begin, end = stack.pop(j), i
                                mention_info = [dataset.docs_flu[doc_num][sent_id][tok_id] for tok_id in
                                                range(begin, end + 1)]
                                mentions.append(Mention(sent_id, begin, end, info=mention_info))
                        elif command:
                            raise ValueError(f"Unknown command '{command}'")
                while len(stack):
                    begin, end = stack.pop(), len(tags) - 1
                    mention_info = [dataset.docs_flu[doc_num][sent_id][tok_id] for tok_id in range(begin, end + 1)]
                    mentions.append(Mention(sent_id, begin, end, info=mention_info))

                # Decode zero mentions
                for i, zdeprel in enumerate(zdeprels):
                    for j in range(self._args.zeros_per_parent):
                        if zdeprel[j] == Dataset.ZDEPREL_PAD or zdeprel[j] == Dataset.ZDEPREL_NONE:
                            break
                        mentions.append(ZeroMention(sent_id, i, -j - 1, zdeprel=self._zdeprels[zdeprel[j]],
                                                    info=[("", "", "PRON")]))

                # Prepare inputs for antecedent prediction
                mentions = sorted(mentions, key=lambda x: (x.begin, -x.end))
                offset = doc_subwords - (word_indices[0] - 2 - tid)
                results[-1].append([]), results_zeros.append([]), b_previous.append([]), b_mentions.append(
                    []), b_refs.append([])
                for doc_mention in doc_mentions:
                    if doc_mention[0] < offset: continue
                    b_previous[-1].append([doc_mention[0] - offset + 1 + tid,
                                           doc_mention[1] if doc_mention[1] < 0 else doc_mention[1] - offset + 1 + tid])
                    b_refs[-1].append(doc_mention[2])
                for mention in mentions:
                    results[-1][-1].append(mention)
                    b_refs[-1].append(mention)
                    b_mentions[-1].append(
                        [word_indices[mention.begin], mention.end if mention.end < 0 else word_indices[mention.end]])
                    doc_mentions.append([doc_subwords + word_indices[mention.begin] - word_indices[0],
                                         mention.end if mention.end < 0 else doc_subwords + word_indices[mention.end] -
                                                                             word_indices[0], mention])
                doc_subwords += word_indices[-1] - word_indices[0]
                sent_id += 1

            # Decode antecedents
            if sum(len(mentions) for mentions in b_mentions) == 0: continue
            b_antecedents = self.compute_antecedents(
                b_embeddings, b_zero_embeddings,
                tf.ragged.constant(b_previous, dtype=tf.int32, ragged_rank=1, inner_shape=(2,)),
                tf.ragged.constant(b_mentions, dtype=tf.int32, ragged_rank=1, inner_shape=(2,)))
            for b in range(b_size):
                len_prev, mentions, refs, antecedents = len(b_previous[b]), b_mentions[b], b_refs[b], b_antecedents[
                    b].numpy()
                for i in range(len(mentions)):
                    j = i - 1
                    while j >= 0 and mentions[j][0] == mentions[i][0]:
                        antecedents[i, j + len_prev] = antecedents[i, i + len_prev] - 1
                        j -= 1
                    j = np.argmax(antecedents[i, :i + len_prev + 1])
                    if j == i + len_prev:
                        refs[i + len_prev].cluster = entities
                        entities += 1
                    else:
                        refs[i + len_prev].cluster = refs[j].cluster

        return results

    # def make_subdocs(self, results, segment_length=5, soft=False):
    #     submaps = [{} for _ in range(len(results))]

    #     for i in range(len(results)):
    #         entities = 0
    #         for j in range(segment_length):
    #             if i + j >= len(results): break
    #             for res in results[i + j]:
    #                 mention = (i + j, res[0], res[1])
    #                 if soft:
    #                     antecedent = None
    #                     for _, candidate in res[2]:
    #                         if candidate[3] >= i:
    #                             antecedent = (candidate[3], candidate[0], candidate[1]) # (sent_id, start_w_id, end_w_id)
    #                             break
    #                     assert not antecedent is None
    #                 else:
    #                     antecedent = (res[2][3], res[2][0], res[2][1]) # (sent_id, start_w_id, end_w_id)
    #                 if antecedent in submaps[i]:
    #                     submaps[i][mention] = submaps[i][antecedent]
    #                 else:
    #                     submaps[i][mention] = entities
    #                     entities += 1

    #     subdocs = [[] for _ in range(len(results))]
    #     for i in range(len(results)):
    #         for mention, cluster in submaps[i].items():
    #             subdocs[i].append(mention + (cluster,))
    #     return subdocs

    def callback(self, epoch: int, datasets: list[tuple[Dataset, tf.data.Dataset]], evaluate: bool) -> None:
        for dataset, pipeline in datasets:
            predicts = self.predict(dataset, pipeline)

            results, zero_results = [], []
            cluster_add = 0
            for doc_i, predict in enumerate(predicts):
                # subdocs = self.make_subdocs(predict, soft=True)
                # subdocs = [[(ment.sent_id, ment.begin, ment.end, ment.cluster) for ment in sent] for sent in predict]
                # for subdoc in subdocs:
                #     for i, mention in enumerate(subdoc):
                #         mention_info = []
                #         if type(mention[2]) == "str": # zero mention
                #             mention_info.append(("", "", "PRON"))
                #         else:
                #             for j in range(mention[1], mention[2] + 1):
                #                 form_lemma_upos = dataset.docs_flu[doc_i][mention[0]][j]
                #                 mention_info.append(form_lemma_upos)

                #         subdoc[i] = (*mention, mention_info)
                clusters = merge_clusters(predict)

                local_results = [[] for _ in range(len(predict))]
                for cluster_id, cluster in enumerate(clusters):
                    for mention in cluster:
                        mention.cluster += cluster_add
                        local_results[mention.sent_id].append(mention)
                        # if isinstance(mention, ZeroMention):

                        # if type(ment[2]) == "str":
                        #     local_zero_results[ment[0]].append((ment[1], ment[2], cluster_id + cluster_add))
                        # else:
                        #     local_results[ment[0]].append((ment[1], ment[2], cluster_id + cluster_add))
                cluster_add += len(clusters)
                for i in range(len(local_results)):
                    local_results[i] = sorted(local_results[i],
                                              key=lambda x: (x.begin, -isinstance(x, ZeroMention), -x.end))
                # for i in range(len(local_zero_results)):
                #     local_zero_results[i] = sorted(local_zero_results[i])

                results.extend(local_results)
                # zero_results.extend(local_zero_results)

            path = os.path.join(self._args.logdir, f"{os.path.splitext(os.path.basename(dataset._path))[0]}.conllu")
            dataset.save_mentions(path, results)  # , zero_results)
            # dataset.save_mentions23(path, results)
            if evaluate:
                os.system(f"run ./corefud-score.sh '{dataset._path}' '{path}'")


def main(params: list[str] | None = None) -> None:
    args = parser.parse_args(params)
    args.depth = 5
    args.encoder = "google/mt5-large"
    args.epochs = 0

    # If supplied, load configuration from a trained model
    if args.load:
        with open(os.path.join(os.path.dirname(args.load[0]), "options.json"), mode="r") as options_file:
            args = argparse.Namespace(**{k: v for k, v in json.load(options_file).items() if k in [
                "batch_size", "depth", "encoder", "right", "segment", "seed", "treebanks", "treebank_id"]})
            args = parser.parse_args(params, namespace=args)
        args.logdir = args.exp if args.exp else os.path.dirname(args.load[0])
    else:
        raise ValueError("--load must be set.")

    # Set the random seed and the number of threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)

    # Load the data
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.encoder)
    tokenizer.add_special_tokens({"additional_special_tokens": [Dataset.TOKEN_EMPTY] +
                                                               [Dataset.TOKEN_TREEBANK.format(i) for i in
                                                                range(len(args.treebanks))] +
                                                               ([
                                                                    Dataset.TOKEN_CLS] if tokenizer.cls_token_id is None and not args.treebank_id else [])})

    if args.dev and args.treebank_id:
        print("When --treebank_id is set and you pass explicit --dev treebanks, they MUST correspond to --treebanks.")
    devs = [Dataset(path.replace("-train.conllu", "-dev.conllu"), tokenizer, args.treebank_id * i)
            for i, path in enumerate([] if args.dev is None else (args.dev or args.treebanks), 1) if path]

    if args.test and args.treebank_id:
        print("When --treebank_id is set and you pass explicit --test treebanks, they MUST correspond to --treebanks.")
    tests = [Dataset(path.replace("-train.conllu", "-test.conllu"), tokenizer, args.treebank_id * i)
             for i, path in enumerate([] if args.test is None else (args.test or args.treebanks), 1) if path]

    with open(os.path.join(os.path.dirname(args.load[0]), "tags.txt"), mode="r") as tags_file:
        tags = [line.rstrip("\r\n") for line in tags_file]
    with open(os.path.join(os.path.dirname(args.load[0]), "zdeprels.txt"), mode="r") as zdeprels_file:
        zdeprels = [line.rstrip("\r\n") for line in zdeprels_file]

    tags_map = {tag: i for i, tag in enumerate(tags)}
    zdeprels_map = {zdeprel: i for i, zdeprel in enumerate(zdeprels)}

    strategy_scope = None
    if len(tf.config.list_physical_devices("GPU")) > 1 and len(args.load) <= 1:
        strategy_scope = tf.distribute.MirroredStrategy().scope()
    with strategy_scope or contextlib.nullcontext():
        # Create pipelines
        devs = [(dev, dev.pipeline(tags_map, zdeprels_map, False, args).ragged_batch(args.batch_size).prefetch(
            tf.data.AUTOTUNE)) for dev in devs]
        tests = [(test, test.pipeline(tags_map, zdeprels_map, False, args).ragged_batch(args.batch_size).prefetch(
            tf.data.AUTOTUNE)) for test in tests]

        model = Model(tokenizer, tags, zdeprels, args)

        if args.dev is not None or args.test is not None:
            os.makedirs(args.logdir, exist_ok=True)
            if args.dev is not None:
                model.callback(0, devs, evaluate=True)
            if args.test is not None:
                model.callback(0, tests, evaluate=False)


if __name__ == "__main__":
    main([] if "__file__" not in globals() else None)
