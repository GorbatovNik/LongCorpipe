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
import asyncio
import contextlib
import datetime
import functools
import json
import os
import pickle
import shutil
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import transformers

import udapi
import udapi.block.corefud.movehead
import udapi.block.corefud.removemisc

parser = argparse.ArgumentParser()
parser.add_argument("--adafactor", default=False, action="store_true", help="Use Adafactor.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--beta_2", default=0.999, type=float, help="Beta2.")
parser.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
parser.add_argument("--depth", default=5, type=int, help="Constrained decoding depth.")
parser.add_argument("--dev", default=None, nargs="*", type=str, help="Predict dev (treebanks).")
parser.add_argument("--encoder", default="google/mt5-large", type=str, help="MLM encoder model.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--exp", default="", type=str, help="Exp name.")
parser.add_argument("--label_smoothing", default=0.2, type=float, help="Label smoothing.")
parser.add_argument("--lazy_adam", default=False, action="store_true", help="Use Lazy Adam.")
parser.add_argument("--learning_rate", default=5e-4, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default=False, action="store_true", help="Decay LR.")
parser.add_argument("--load", default=[], type=str, nargs="*", help="Models to load.")
parser.add_argument("--max_links", default=None, type=int, help="Max antecedent links to train on.")
parser.add_argument("--resample", default=[], nargs="*", type=float, help="Train data resample ratio.")
parser.add_argument("--right", default=50, type=int, help="Reserved space for right context, if any.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--segment", default=512, type=int, help="Segment size")
parser.add_argument("--test", default=None, nargs="*", type=str, help="Predict test (treebanks).")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train", default=False, action="store_true", help="Perform training.")
parser.add_argument("--treebanks", default=[], nargs="+", type=str, help="Data.")
parser.add_argument("--treebank_id", default=False, action="store_true", help="Use treebank id.")
parser.add_argument("--warmup", default=0.1, type=float, help="Warmup ratio.")
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
        if treebank_id: # 0 is deliberately considered as no treebank id
            treebank_token = tokenizer.vocab[self.TOKEN_TREEBANK.format(treebank_id - 1)]
            if self._cls is None:
                self._cls = treebank_token
            else:
                self._treebank_token = [treebank_token]
        if self._cls is None:
            self._cls = tokenizer.vocab[self.TOKEN_CLS]

        # Create the tokenized documents if they do not exist
        cache_path = f"{path}.mentions.{os.path.basename(tokenizer.name_or_path)}"
        if not os.path.exists(cache_path) or os.path.getmtime(cache_path) <= os.path.getmtime(path):
            # Parse with Udapi
            if not os.path.exists(f"{path}.mentions") or os.path.getmtime(f"{path}.mentions") <= os.path.getmtime(path):
                docs, new_doc = [], []
                for doc in udapi.block.read.conllu.Conllu(files=[path]).read_documents():
                    for tree in doc.trees:
                        if tree.newdoc is not None and new_doc:
                            docs.append(new_doc)
                            new_doc = []
                        words, coref_mentions = [], set()
                        for node in tree.descendants:
                            words.append(node.form)
                            coref_mentions.update(node.coref_mentions)
                        for enode in tree.empty_nodes:
                            coref_mentions.update(enode.coref_mentions)

                        dense_mentions = []
                        for mention in [mention for mention in coref_mentions if not mention.head.is_empty()]:
                            span = [word for word in mention.words if not word.is_empty()]
                            start = end = span.index(mention.head)
                            while start > 0 and span[start - 1].ord + 1 == span[start].ord: start -= 1
                            while end < len(span) - 1 and span[end].ord + 1== span[end + 1].ord: end += 1
                            dense_mentions.append(((span[start].ord - 1, span[end].ord - 1), mention.entity.eid, start > 0 or end + 1 < len(span)))
                        dense_mentions = sorted(dense_mentions, key=lambda x:(x[0][0], -x[0][1], x[2]))

                        mentions = []
                        for i, mention in enumerate(dense_mentions):
                            if i and dense_mentions[i-1][0] == mention[0]:
                                print(f"Multiple same mentions {mention[2]}/{dense_mentions[i-1][2]} in sent_id {tree.sent_id}: {tree.get_sentence()}", flush=True)
                                continue
                            mentions.append((mention[0][0], mention[0][1], mention[1]))

                        zero_mentions = []
                        for mention in [mention for mention in coref_mentions if mention.head.is_empty()]:
                            if len(mention.words) > 1:
                                print(f"A empty-node-head mention with multiple words {mention.words} in sent_id {tree.sent_id}: {tree.get_sentence()}", flush=True)
                            assert len(mention.head.deps) >= 1
                            zero_mentions.append((mention.head.deps[0]["parent"].ord - 1, mention.head.deps[0]["deprel"], mention.entity.eid))
                        zero_mentions = sorted(zero_mentions)
                        new_doc.append((words, mentions, zero_mentions))
                if new_doc:
                    docs.append(new_doc)
                with open(f"{path}.mentions", "wb") as cache_file:
                    pickle.dump(docs, cache_file, protocol=3)
            with open(f"{path}.mentions", "rb") as cache_file:
                docs = pickle.load(cache_file)

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
                        if subword[0] == 6 and "xlm-r" in tokenizer.name_or_path: # Hack: remove the space-only token in XLM-R
                            subword = subword[1:]
                        assert len(subword) > 0
                        subwords.extend(subword)

                        tag = [str(len(stack))]
                        for _ in range(2):
                            for j in reversed(range(len(stack))):
                                start, end, eid = stack[j]
                                if end == i:
                                    tag.append(f"POP:{len(stack)-j}")
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
                    subword_mentions = sorted(subword_mentions, key=lambda x:(x[0], -x[1]))
                    new_doc.append((subwords, word_indices, word_tags, word_zdeprels, subword_mentions))
                self.docs.append(new_doc)

            with open(cache_path, "wb") as cache_file:
                pickle.dump(self.docs, cache_file, protocol=3)
        with open(cache_path, "rb") as cache_file:
            self.docs = pickle.load(cache_file)
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

    def pipeline(self, tags_map: dict[str, int], zdeprels_map: dict[str, int], train: bool, args: argparse.Namespace) -> tf.data.Dataset:
        def generator():
            tid = len(self._treebank_token)
            for doc in self.docs:
                p_subwords, p_subword_mentions = [], []
                for doc_i, (subwords, word_indices, word_tags, word_zdeprels, subword_mentions) in enumerate(doc):
                    subword_mentions = [(s, e, eid) for s, e, eid in subword_mentions if e >= -args.zeros_per_parent]
                    if not train and len(subwords) + 4 + tid > args.segment:
                        print("Truncating a long sentence during prediction")
                        subwords = subwords[:args.segment - 4 - tid]
                    assert train or len(subwords) + 4 + tid <= args.segment
                    if len(subwords) + 4 + tid <= args.segment:
                        right_reserve = min((args.segment - 4 - tid - len(subwords)) // 2, args.right or 0)
                        context = min(args.segment - 4 - tid - len(subwords) - right_reserve, len(p_subwords))
                        word_indices = [context + 2 + tid + i for i in word_indices + [len(subwords)]]
                        e_subwords = [self._cls, *self._treebank_token, *p_subwords[-context:], self._sep, *subwords, self._sep]
                        if args.right is not None:
                            i = doc_i + 1
                            while i < len(doc) and len(e_subwords) + 1 < args.segment:
                                e_subwords.extend(doc[i][0][:args.segment - len(e_subwords) - 1])
                                i += 1
                        e_subwords.append(self._sep)

                        output = (e_subwords, word_indices)
                        if train:
                            offset = len(p_subwords) - context
                            prev = [(s - offset + 1 + tid, e if e < 0 else e - offset + 1 + tid, eid) for s, e, eid in p_subword_mentions if s >= offset]
                            prev_pos = np.array([[s, e] for s, e, _ in prev], dtype=np.int32).reshape([-1, 2])
                            prev_eid = np.array([eid for _, _, eid in prev], dtype=str)
                            ment = [(context + 2 + tid + s, e if e < 0 else context + 2 + tid + e, eid) for s, e, eid in subword_mentions]
                            ment_pos = np.array([[s, e] for s, e, _ in ment], dtype=np.int32).reshape([-1, 2])
                            ment_eid = np.array([eid for _, _, eid in ment], dtype=str)
                            mask = ment_pos[:, 0, None] > np.concatenate([prev_pos[:, 0], ment_pos[:, 0]])[None, :]
                            diag = np.pad(np.eye(len(ment_pos)), [[0, 0], [len(prev_pos), 0]])
                            gold = (ment_eid[:, None] == np.concatenate([prev_eid, ment_eid])[None, :]) * mask
                            gold = np.where(np.sum(gold, axis=1, keepdims=True) > 0, gold, diag)
                            if args.max_links is not None:
                                max_link_mask = np.cumsum(gold, axis=1)
                                gold *= (max_link_mask > max_link_mask[:, -1:] - args.max_links)
                            gold /= np.sum(gold, axis=1, keepdims=True)
                            mask = mask + diag
                            if args.label_smoothing:
                                gold = (1 - args.label_smoothing) * gold + args.label_smoothing * (mask / np.sum(mask, axis=1, keepdims=True))

                            word_tags = [tags_map[tag] for tag in word_tags]
                            word_zdeprels_padded = np.zeros([len(word_tags), args.zeros_per_parent], np.int32)
                            for zdeprels_padded, zdeprels in zip(word_zdeprels_padded, word_zdeprels):
                                zdeprels_padded[:min(args.zeros_per_parent, len(zdeprels) + 1)] = (
                                    [zdeprels_map[zdeprel] for zdeprel in zdeprels] + [self.ZDEPREL_NONE])[:args.zeros_per_parent]

                            output = (output, (word_tags, word_zdeprels_padded, prev_pos, ment_pos, mask, gold))
                        yield output

                    p_subword_mentions.extend((s + len(p_subwords), e if e < 0 else e + len(p_subwords), eid) for s, e, eid in subword_mentions)
                    p_subwords.extend(subwords)

        output_signature=(tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32))
        if train:
            output_signature = (output_signature, (
                tf.TensorSpec([None], tf.int32), tf.TensorSpec([None, args.zeros_per_parent], tf.int32), tf.TensorSpec([None, 2], tf.int32),
                tf.TensorSpec([None, 2], tf.int32), tf.TensorSpec([None, None], tf.bool), tf.TensorSpec([None, None], tf.float32),
            ))

        pipeline = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        pipeline = pipeline.cache()
        pipeline = pipeline.apply(tf.data.experimental.assert_cardinality(sum(1 for _ in pipeline)))
        return pipeline

    def save_mentions(self, path: str, mentions: list[list[tuple[int, int, int]]], zero_mentions: list[list[tuple[int, str, int]]]) -> None:
        doc = udapi.block.read.conllu.Conllu(files=[self._path]).read_documents()[0]
        udapi.block.corefud.removemisc.RemoveMisc(attrnames="Entity,SplitAnte,Bridge").apply_on_document(doc)

        entities = {}
        for i, tree in enumerate(doc.trees):
            tree.empty_nodes = []  # Drop existing empty nodes
            for node in tree.descendants:  # Remove references to empty nodes also from DEPS, by replacing them by the main dependency edge
                if "." in node.raw_deps:
                    node.raw_deps = f"{node.parent.ord}:{node.deprel}"
            ords = {}
            for parent, deprel, eid in zero_mentions[i]:  # Add predicted empty nodes
                tree.create_empty_child()
                ords[parent] = ords.get(parent, 0) + 1
                tree.empty_nodes[-1].ord = f"{parent+1}.{ords[parent]}"
                tree.empty_nodes[-1].raw_deps = f"{parent+1}:{deprel}"
                if not eid in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention([tree.empty_nodes[-1]], entity=entities[eid])
            nodes = tree.descendants_and_empty
            for start, end, eid in mentions[i]:
                if not eid in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention([node for node in nodes if start <= node.ord - 1 <= end], entity=entities[eid])
        doc._eid_to_entity = {entity._eid: entity for entity in sorted(entities.values())}
        udapi.block.corefud.movehead.MoveHead(bugs='ignore').apply_on_document(doc)
        udapi.block.write.conllu.Conllu(files=[path]).apply_on_document(doc)


class Model(tf.keras.Model):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, tags: list[str], zdeprels: list[str], args: argparse.Namespace) -> None:
        super().__init__()
        self._tags = tags
        self._zdeprels = zdeprels
        self._args = args

        assert tags[0] == "" # Used as a boundary tag in CRF
        self._allowed_tag_transitions = tf.constant(Dataset.allowed_tag_transitions(tags, args.depth + 1))
        self._boundary_logits = tf.cast(tf.range(self._allowed_tag_transitions.shape[0]) > 0, tf.float32) * -1e6

        self._encoder = transformers.TFMT5EncoderModel if "mt5" in args.encoder.lower() else transformers.TFAutoModel
        if not args.load:
            self._encoder = self._encoder.from_pretrained(
                args.encoder, from_pt=any(m in args.encoder.lower() for m in ["rubert", "herbert", "flaubert", "litlat", "roberta-base-ca", "spanbert", "xlm-v", "infoxlm"]))
        else:
            self._encoder = self._encoder.from_config(transformers.AutoConfig.from_pretrained(args.encoder))
            self._encoder(tf.constant([[0]], dtype=tf.int32), attention_mask=tf.constant([[1.]], dtype=tf.float32))
        self._encoder.resize_token_embeddings(len(tokenizer.vocab))
        self._dense_hidden_q = tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu, name="dense_hidden_q")
        self._dense_hidden_k = tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu, name="dense_hidden_k")
        self._dense_hidden_tags = tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu, name="dense_hidden_tags")
        self._dense_q = tf.keras.layers.Dense(self._encoder.config.hidden_size, use_bias=False, name="dens_q")
        self._dense_k = tf.keras.layers.Dense(self._encoder.config.hidden_size, use_bias=False, name="dens_k")
        self._dense_tags = tf.keras.layers.Dense(len(tags), name="dense_tags")
        self._dense_hidden_zeros = [
            tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu, name=f"dense_hidden_zeros_{i}") for i in range(args.zeros_per_parent)]
        self._dense_zeros = [tf.keras.layers.Dense(self._encoder.config.hidden_size, name=f"dense_zeros_{i}") for i in range(args.zeros_per_parent)]
        self._dense_hidden_zdeprels = [
            tf.keras.layers.Dense(4 * self._encoder.config.hidden_size, activation=tf.nn.relu, name=f"dense_hidden_zdeprels_{i}") for i in range(args.zeros_per_parent)]
        self._dense_zdeprels = [tf.keras.layers.Dense(len(zdeprels), name=f"dense_zdeprels_{i}") for i in range(args.zeros_per_parent)]

        if args.load:
            self.compute_tags(tf.ragged.constant([[0]]), tf.ragged.constant([[0]]))
            self.compute_antecedents(tf.zeros([1, 1, self._encoder.config.hidden_size]), tf.zeros([1, args.zeros_per_parent, self._encoder.config.hidden_size]),
                                     *[tf.ragged.constant([[[0, 0]]], dtype=tf.int32, ragged_rank=1, inner_shape=(2,))] * 2)
            self.built = True
            self.load_weights(args.load[0])

    def compile(self, train: tf.data.Dataset) -> None:
        args = self._args
        warmup_steps = int(args.warmup * args.epochs * len(train))
        learning_rate = tf.optimizers.schedules.PolynomialDecay(
            args.learning_rate, args.epochs * len(train) - warmup_steps, 0. if args.learning_rate_decay else args.learning_rate)
        if warmup_steps:
            class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
                def __init__(self, warmup_steps, following_schedule):
                    self._warmup_steps = warmup_steps
                    self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
                    self._following = following_schedule
                def __call__(self, step):
                    return tf.cond(step < self._warmup_steps,
                                   lambda: self._warmup(step),
                                   lambda: self._following(step - self._warmup_steps))
            learning_rate = LinearWarmup(warmup_steps, learning_rate)
        if args.adafactor:
            optimizer = tf.optimizers.Adafactor(learning_rate=learning_rate)
        elif args.lazy_adam:
            optimizer = tfa.optimizers.LazyAdam(learning_rate=learning_rate, beta_2=args.beta_2)
        else:
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_2=args.beta_2)
        super().compile(optimizer=optimizer)

    def crf_decode(self, logits: tf.RaggedTensor, crf_weights: tf.Tensor) -> tf.RaggedTensor:
        boundary_logits = tf.broadcast_to(self._boundary_logits, [logits.bounding_shape(0), 1, len(self._boundary_logits)])
        logits = tf.concat([boundary_logits, logits, boundary_logits], axis=1)
        predictions, _ = tfa.text.crf_decode(logits.to_tensor(), crf_weights, logits.row_lengths())
        predictions = tf.RaggedTensor.from_tensor(predictions, logits.row_lengths())
        predictions = predictions[:, 1:-1]
        return predictions

    @tf.function(experimental_relax_shapes=True)
    def compute_tags(self, subwords, word_indices, training=False) -> tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
        if training or subwords.bounding_shape(0) > 0:
            embeddings = self._encoder(subwords.to_tensor(), attention_mask=tf.sequence_mask(subwords.row_lengths(), dtype=tf.float32),
                                       training=training).last_hidden_state
        else:
            # During prediction, we need to correctly handle batches of size 0 when using multiple GPUs
            embeddings = tf.zeros([0, 0, self._encoder.config.hidden_size])
        words = tf.gather(embeddings, word_indices[:, :-1], batch_dims=1)
        tag_logits = self._dense_tags(self._dense_hidden_tags(words))

        zero_embeddings = []
        for i in range(self._args.zeros_per_parent):
            zero_embeddings.append(self._dense_zeros[i](self._dense_hidden_zeros[i](tf.concat([embeddings] + zero_embeddings, axis=-1))))
        zdeprel_logits = []
        for i in range(self._args.zeros_per_parent):
            zdeprel_logits.append(self._dense_zdeprels[i](self._dense_hidden_zdeprels[i](tf.gather(zero_embeddings[i], word_indices[:, :-1], batch_dims=1))))
        return embeddings, tf.concat(zero_embeddings, axis=1), tag_logits, tf.stack(zdeprel_logits, axis=-2)

    @tf.function(experimental_relax_shapes=True)
    def compute_antecedents(self, embeddings, zero_embeddings, previous, mentions) -> tf.RaggedTensor:
        mentions_embedded = tf.gather(embeddings, tf.math.maximum(mentions, 0), batch_dims=1).values
        mentions_embedded = tf.reshape(mentions_embedded, [-1, np.prod(mentions_embedded.shape[-2:])])
        zero_mentions_embedded = tf.gather(zero_embeddings, self._args.zeros_per_parent * mentions[..., 0] + tf.math.maximum(-mentions[..., 1] - 1, 0), batch_dims=1).values
        zero_mentions_embedded = tf.tile(zero_mentions_embedded, [1, 2])
        mentions_embedded = tf.where(mentions[..., 1:].values >= 0, mentions_embedded, zero_mentions_embedded)
        queries = mentions.with_values(self._dense_q(self._dense_hidden_q(mentions_embedded)))
        keys_mentions = mentions.with_values(self._dense_k(self._dense_hidden_k(mentions_embedded)))

        previous_embedded = tf.gather(embeddings, tf.math.maximum(previous, 0), batch_dims=1).values
        previous_embedded = tf.reshape(previous_embedded, [-1, mentions_embedded.shape[-1]])
        zero_previous_embedded = tf.gather(zero_embeddings, self._args.zeros_per_parent * previous[..., 0] + tf.math.maximum(-previous[..., 1] - 1, 0), batch_dims=1).values
        zero_previous_embedded = tf.tile(zero_previous_embedded, [1, 2])
        previous_embedded = tf.where(previous[..., 1:].values >= 0, previous_embedded, zero_previous_embedded)
        keys_previous = previous.with_values(self._dense_k(self._dense_hidden_k(previous_embedded)))
        keys = tf.concat([keys_previous, keys_mentions], axis=1)
        weights = tf.matmul(queries.to_tensor(), keys.to_tensor(), transpose_b=True) / (self._dense_q.units ** 0.5)
        return weights

    def train_step(self, data: tuple) -> dict[str, tf.Tensor]:
        (subwords, word_indices), (tags, zdeprels, previous, mentions, mask, antecedents) = data
        with tf.GradientTape() as tape:
            # Tagging part
            embeddings, zero_embeddings, tag_logits, zdeprel_logits = self.compute_tags(subwords, word_indices, training=True)
            tags_loss = tf.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=self._args.label_smoothing, reduction=tf.losses.Reduction.SUM)(
                    tf.one_hot(tags.values, len(self._tags)), tag_logits.values) / tf.cast(tf.shape(tag_logits.values)[0], tf.float32)
            zdeprels_mask = tf.cast(zdeprels.values != Dataset.ZDEPREL_PAD, tf.float32)
            zdeprels_loss = tf.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=self._args.label_smoothing, reduction=tf.losses.Reduction.SUM)(
                    tf.one_hot(zdeprels.values, len(self._zdeprels)), zdeprel_logits.values, zdeprels_mask) / tf.math.reduce_sum(zdeprels_mask)

            # Antecedents part
            def antecedent_loss():
                weights = self.compute_antecedents(embeddings, zero_embeddings, previous, mentions)
                mask_dense = tf.cast(mask.to_tensor(), tf.float32)
                weights = weights[:, :, :tf.shape(mask_dense)[-1]] # Happens when the largest number of mentions have 0 queries
                weights = mask_dense * weights + (1 - mask_dense) * -1e9
                return tf.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.SUM)(
                    antecedents.values.to_tensor(), tf.RaggedTensor.from_tensor(weights, antecedents.row_lengths()).values
                ) / tf.cast(tf.math.reduce_sum(antecedents.row_lengths()), tf.float32)
            antecedent_loss = tf.cond(tf.math.reduce_sum(antecedents.row_lengths()) != 0, antecedent_loss, lambda: 0.)

            loss = tags_loss + zdeprels_loss + antecedent_loss

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return {"tags_loss": tags_loss, "zdeprels_loss": zdeprels_loss, "antecedent_loss": antecedent_loss, "loss": loss,
                "lr": self.optimizer.learning_rate(self.optimizer.iterations)
                if callable(self.optimizer.learning_rate) else self.optimizer.learning_rate}

    def predict(self, dataset: Dataset, pipeline: tf.data.Dataset) -> tuple[list[list[tuple[int, int, int]]], list[list[tuple[int, str, int]]]]:
        tid = len(dataset._treebank_token)

        results, results_zeros, entities = [], [], 0
        doc_mentions, doc_subwords = [], 0
        for b_subwords, b_word_indices in pipeline:
            b_embeddings, b_zero_embeddings, b_tag_logits, b_zdeprel_logits = self.compute_tags(b_subwords, b_word_indices)
            b_size = b_word_indices.shape[0]
            b_tag_logits = b_tag_logits.with_values(tf.math.log_softmax(tf.tile(b_tag_logits.values, [1, self._args.depth + 1]), axis=-1))
            b_tags = self.crf_decode(b_tag_logits, (1 - self._allowed_tag_transitions) * -1e6)
            b_zdeprels = b_zdeprel_logits.with_values(tf.argmax(b_zdeprel_logits.values, axis=-1))

            b_previous, b_mentions, b_refs = [], [], []
            for b in range(b_size):
                word_indices, tags, zdeprels = b_word_indices[b].numpy(), b_tags[b].numpy(), b_zdeprels[b].numpy()
                if word_indices[0] == 2 + tid:
                    doc_mentions, doc_subwords = [], 0

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
                                mentions.append((stack.pop(j), i, None))
                        elif command:
                            raise ValueError(f"Unknown command '{command}'")
                while len(stack):
                    mentions.append((stack.pop(), len(tags) - 1, None))

                # Decode zero mentions
                for i, zdeprel in enumerate(zdeprels):
                    for j in range(self._args.zeros_per_parent):
                        if zdeprel[j] == Dataset.ZDEPREL_PAD or zdeprel[j] == Dataset.ZDEPREL_NONE:
                            break
                        mentions.append((i, -j - 1, self._zdeprels[zdeprel[j]]))

                # Prepare inputs for antecedent prediction
                mentions = sorted(set(mentions), key=lambda x: (x[0], -x[1]))
                offset = doc_subwords - (word_indices[0] - 2 - tid)
                results.append([]), results_zeros.append([]), b_previous.append([]), b_mentions.append([]), b_refs.append([])
                for doc_mention in doc_mentions:
                    if doc_mention[0] < offset: continue
                    b_previous[-1].append([doc_mention[0] - offset + 1 + tid, doc_mention[1] if doc_mention[1] < 0 else doc_mention[1] - offset + 1 + tid])
                    b_refs[-1].append(doc_mention[2])
                for mention in mentions:
                    if mention[2] is not None:
                        result_mention = [mention[0], mention[2], None]
                        results_zeros[-1].append(result_mention)
                    else:
                        result_mention = [mention[0], mention[1], None]
                        results[-1].append(result_mention)
                    b_refs[-1].append(result_mention)
                    b_mentions[-1].append([word_indices[mention[0]], mention[1] if mention[1] < 0 else word_indices[mention[1]]])
                    doc_mentions.append([doc_subwords + word_indices[mention[0]] - word_indices[0],
                                         mention[1] if mention[1] < 0 else doc_subwords + word_indices[mention[1]] - word_indices[0], result_mention])
                doc_subwords += word_indices[-1] - word_indices[0]

            # Decode antecedents
            if sum(len(mentions) for mentions in b_mentions) == 0: continue
            b_antecedents = self.compute_antecedents(
                b_embeddings, b_zero_embeddings, tf.ragged.constant(b_previous, dtype=tf.int32, ragged_rank=1, inner_shape=(2,)),
                tf.ragged.constant(b_mentions, dtype=tf.int32, ragged_rank=1, inner_shape=(2,)))
            for b in range(b_size):
                len_prev, mentions, refs, antecedents = len(b_previous[b]), b_mentions[b], b_refs[b], b_antecedents[b].numpy()
                for i in range(len(mentions)):
                    j = i - 1
                    while j >= 0 and mentions[j][0] == mentions[i][0]:
                        antecedents[i, j + len_prev] = antecedents[i, i + len_prev] - 1
                        j -= 1
                    j = np.argmax(antecedents[i, :i + len_prev + 1])
                    if j == i + len_prev:
                        entities += 1
                        refs[i + len_prev][2] = entities
                    else:
                        refs[i + len_prev][2] = refs[j][2]

        return results, results_zeros

    def callback(self, epoch: int, datasets: list[tuple[Dataset, tf.data.Dataset]], evaluate: bool) -> None:
        for dataset, pipeline in datasets:
            mentions, zero_mentions = self.predict(dataset, pipeline)
            path = os.path.join(self._args.logdir, f"{os.path.splitext(os.path.basename(dataset._path))[0]}.{epoch:02d}.conllu")
            dataset.save_mentions(path, mentions, zero_mentions)
            if evaluate:
                os.system(f"sbatch -p cpu-troja -o /dev/null run ./corefud-score.sh '{dataset._path}' '{path}'")


class ModelEnsemble:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, tags: list[str], zdeprels: list[str], args: argparse.Namespace) -> None:
        assert len(tf.config.list_physical_devices("GPU")) >= len (args.load)

        self._tags = tags
        self._zdeprels = zdeprels
        self._args = args
        self._models = []
        for i, model in enumerate(args.load):
            with open(os.path.join(os.path.dirname(model), "options.json"), mode="r") as options_file:
                model_args = argparse.Namespace(**vars(args))
                model_args.load = [model]
                model_args.encoder = json.load(options_file)["encoder"]
            with tf.device(f"/gpu:{i}"):
                self._models.append(Model(tokenizer, tags, zdeprels, model_args))

    def np_softmax(self, x):
        x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return x / np.sum(x, axis=-1, keepdims=True)

    def predict(self, dataset: Dataset, pipeline: tf.data.Dataset) -> tuple[list[list[tuple[int, int, int]]], list[list[tuple[int, str, int]]]]:
        tid = len(dataset._treebank_token)

        results, results_zeros, entities = [], [], 0
        doc_mentions, doc_subwords = [], 0
        for b_subwords, b_word_indices in pipeline:
            async def do_compute_tags():
                def compute_tags(i):
                    with tf.device(f"/gpu:{i}"):
                        return self._models[i].compute_tags(b_subwords, b_word_indices)
                return await asyncio.gather(*[asyncio.to_thread(compute_tags, i) for i in range(len(self._models))])
            b_embeddings, b_zero_embeddings, b_tag_logits, b_zdeprel_logits = zip(*asyncio.run(do_compute_tags()))
            b_tag_logits = tf.math.log(sum(tf.nn.softmax(logits, axis=-1) for logits in b_tag_logits))
            b_zdeprel_logits = sum(logits for logits in b_zdeprel_logits)
            b_size = b_word_indices.shape[0]
            b_tag_logits = b_tag_logits.with_values(tf.math.log_softmax(tf.tile(b_tag_logits.values, [1, self._args.depth + 1]), axis=-1))
            b_tags = self._models[0].crf_decode(b_tag_logits, (1 - self._models[0]._allowed_tag_transitions) * -1e6)
            b_zdeprels = b_zdeprel_logits.with_values(tf.argmax(b_zdeprel_logits.values, axis=-1))

            b_previous, b_mentions, b_refs = [], [], []
            for b in range(b_size):
                word_indices, tags, zdeprels = b_word_indices[b].numpy(), b_tags[b].numpy(), b_zdeprels[b].numpy()
                if word_indices[0] == 2 + tid:
                    doc_mentions, doc_subwords = [], 0

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
                                mentions.append((stack.pop(j), i, None))
                        elif command:
                            raise ValueError(f"Unknown command '{command}'")
                while len(stack):
                    mentions.append((stack.pop(), len(tags) - 1, None))

                # Decode zero mentions
                for i, zdeprel in enumerate(zdeprels):
                    for j in range(self._args.zeros_per_parent):
                        if zdeprel[j] == Dataset.ZDEPREL_PAD or zdeprel[j] == Dataset.ZDEPREL_NONE:
                            break
                        mentions.append((i, -j - 1, self._zdeprels[zdeprel[j]]))

                # Prepare inputs for antecedent prediction
                mentions = sorted(set(mentions), key=lambda x: (x[0], -x[1]))
                offset = doc_subwords - (word_indices[0] - 2 - tid)
                results.append([]), results_zeros.append([]), b_previous.append([]), b_mentions.append([]), b_refs.append([])
                for doc_mention in doc_mentions:
                    if doc_mention[0] < offset: continue
                    b_previous[-1].append([doc_mention[0] - offset + 1 + tid, doc_mention[1] if doc_mention[1] < 0 else doc_mention[1] - offset + 1 + tid])
                    b_refs[-1].append(doc_mention[2])
                for mention in mentions:
                    if mention[2] is not None:
                        result_mention = [mention[0], mention[2], None]
                        results_zeros[-1].append(result_mention)
                    else:
                        result_mention = [mention[0], mention[1], None]
                        results[-1].append(result_mention)
                    b_refs[-1].append(result_mention)
                    b_mentions[-1].append([word_indices[mention[0]], mention[1] if mention[1] < 0 else word_indices[mention[1]]])
                    doc_mentions.append([doc_subwords + word_indices[mention[0]] - word_indices[0],
                                         mention[1] if mention[1] < 0 else doc_subwords + word_indices[mention[1]] - word_indices[0], result_mention])
                doc_subwords += word_indices[-1] - word_indices[0]

            # Decode antecedents
            if sum(len(mentions) for mentions in b_mentions) == 0: continue
            async def do_compute_antecedents():
                def compute_antecedents(i):
                    with tf.device(f"/gpu:{i}"):
                        return self._models[i].compute_antecedents(
                            b_embeddings[i], b_zero_embeddings[i], tf.ragged.constant(b_previous, dtype=tf.int32, ragged_rank=1, inner_shape=(2,)),
                            tf.ragged.constant(b_mentions, dtype=tf.int32, ragged_rank=1, inner_shape=(2,)))
                return await asyncio.gather(*[asyncio.to_thread(compute_antecedents, i) for i in range(len(self._models))])
            b_antecedents = asyncio.run(do_compute_antecedents())
            for b in range(b_size):
                len_prev, mentions, refs, antecedents = len(b_previous[b]), b_mentions[b], b_refs[b], [a[b].numpy() for a in b_antecedents]
                for i in range(len(mentions)):
                    j = i - 1
                    while j >= 0 and mentions[j][0] == mentions[i][0]:
                        for a in antecedents:
                            a[i, j + len_prev] = a[i, i + len_prev] - 1
                        j -= 1
                    j = np.argmax(sum(self.np_softmax(a[i, :i + len_prev + 1]) for a in antecedents))
                    if j == i + len_prev:
                        entities += 1
                        refs[i + len_prev][2] = entities
                    else:
                        refs[i + len_prev][2] = refs[j][2]

        return results, results_zeros

    def callback(self, epoch: int, datasets: list[tuple[Dataset, tf.data.Dataset]], evaluate: bool) -> None:
        return Model.callback(self, epoch, datasets, evaluate)


def main(params: list[str] | None = None) -> None:
    args = parser.parse_args(params)

    # If supplied, load configuration from a trained model
    if args.load:
        with open(os.path.join(os.path.dirname(args.load[0]), "options.json"), mode="r") as options_file:
            args = argparse.Namespace(**{k: v for k, v in json.load(options_file).items() if k in [
                "batch_size", "depth", "encoder", "right", "segment", "seed", "treebanks", "treebank_id"]})
            args = parser.parse_args(params, namespace=args)
        args.logdir = args.exp if args.exp else os.path.dirname(args.load[0])
    else:
        if not args.train:
            raise ValueError("Either --load or --train must be set.")
        args.logdir = os.path.join("logs", "{}{}-{}-{}-{}".format(
            args.exp + (args.exp and "-"),
            os.path.splitext(os.path.basename(globals().get("__file__", "notebook")))[0],
            os.environ.get("SLURM_JOB_ID", ""),
            datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
            ",".join(("{}={}".format(
                re.sub("(.)[^_]*_?", r"\1", k),
                ",".join(re.sub(r"^.*/", "", str(x)) for x in ((v if len(v) <= 1 else [v[0], "..."]) if isinstance(v, list) else [v])),
            ) for k, v in sorted(vars(args).items()) if k not in ["debug", "exp", "load", "threads"]))
        ))
        print(json.dumps(vars(args), sort_keys=True, ensure_ascii=False, indent=2))

    # Set the random seed and the number of threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)

    # Load the data
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.encoder)
    tokenizer.add_special_tokens({"additional_special_tokens": [Dataset.TOKEN_EMPTY] +
                                  [Dataset.TOKEN_TREEBANK.format(i) for i in range(len(args.treebanks))] +
                                  ([Dataset.TOKEN_CLS] if tokenizer.cls_token_id is None and not args.treebank_id else [])})


    trains = [Dataset(path, tokenizer, args.treebank_id * i) for i, path in enumerate(args.treebanks, 1)] if args.train else []

    if args.dev and args.treebank_id:
        print("When --treebank_id is set and you pass explicit --dev treebanks, they MUST correspond to --treebanks.")
    devs = [Dataset(path.replace("-train.conllu", "-dev.conllu"), tokenizer, args.treebank_id * i)
            for i, path in enumerate([] if args.dev is None else (args.dev or args.treebanks), 1) if path]

    if args.test and args.treebank_id:
        print("When --treebank_id is set and you pass explicit --test treebanks, they MUST correspond to --treebanks.")
    tests = [Dataset(path.replace("-train.conllu", "-test.conllu"), tokenizer, args.treebank_id * i)
             for i, path in enumerate([] if args.test is None else (args.test or args.treebanks), 1) if path]

    if args.load:
        with open(os.path.join(os.path.dirname(args.load[0]), "tags.txt"), mode="r") as tags_file:
            tags = [line.rstrip("\r\n") for line in tags_file]
        with open(os.path.join(os.path.dirname(args.load[0]), "zdeprels.txt"), mode="r") as zdeprels_file:
            zdeprels = [line.rstrip("\r\n") for line in zdeprels_file]
    else:
        tags = Dataset.create_tags(trains)
        zdeprels = Dataset.create_zdeprels(trains)
    tags_map = {tag: i for i, tag in enumerate(tags)}
    zdeprels_map = {zdeprel: i for i, zdeprel in enumerate(zdeprels)}

    strategy_scope = None
    if len(tf.config.list_physical_devices("GPU")) > 1 and len(args.load) <= 1:
        strategy_scope = tf.distribute.MirroredStrategy().scope()
    with strategy_scope or contextlib.nullcontext():
        # Create pipelines
        if args.train:
            trains = [train.pipeline(tags_map, zdeprels_map, True, args) for train in trains]
            if args.resample:
                steps, *ratios = args.resample
                assert len(ratios) == len(trains)
                ratios = [ratio / sum(ratios) for ratio in ratios]
                trains = [train.shuffle(len(train)).repeat().take(1 + int(steps * args.batch_size * ratio))
                          for train, ratio in zip(trains, ratios)]
            train = functools.reduce(lambda x, y: x.concatenate(y), trains)
            train = train.shuffle(len(train), seed=args.seed).ragged_batch(args.batch_size, True).prefetch(tf.data.AUTOTUNE)
        devs = [(dev, dev.pipeline(tags_map, zdeprels_map, False, args).ragged_batch(args.batch_size).prefetch(tf.data.AUTOTUNE)) for dev in devs]
        tests = [(test, test.pipeline(tags_map, zdeprels_map, False, args).ragged_batch(args.batch_size).prefetch(tf.data.AUTOTUNE)) for test in tests]

        model = (ModelEnsemble if len(args.load) > 1 else Model)(tokenizer, tags, zdeprels, args)

        if args.train:
            # Create logdir with the source, options, and tags
            os.makedirs(args.logdir)
            shutil.copy2(__file__, os.path.join(args.logdir, os.path.basename(__file__)))
            with open(os.path.join(args.logdir, "options.json"), "w") as json_file:
                json.dump(vars(args), json_file, sort_keys=True, ensure_ascii=False, indent=2)
            with open(os.path.join(args.logdir, "tags.txt"), "w") as tags_file:
                for tag in tags:
                    print(tag, file=tags_file)
            with open(os.path.join(args.logdir, "zdeprels.txt"), "w") as zdeprels_file:
                for zdeprel in zdeprels:
                    print(zdeprel, file=zdeprels_file)

            # Compile the model and train
            model.compile(train)
            model.fit(train, epochs=args.epochs, verbose=int(os.environ.get("VERBOSE", "2")), callbacks=[
                tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, _: model.save_weights(f"{args.logdir}/model{epoch+1:02d}.h5")),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, _: model.callback(epoch + 1, devs, evaluate=True)),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, _: model.callback(epoch + 1, tests, evaluate=False)),
            ])
        elif args.dev is not None or args.test is not None:
            os.makedirs(args.logdir, exist_ok=True)
            if args.dev is not None:
                model.callback(args.epochs, devs, evaluate=True)
            if args.test is not None:
                model.callback(args.epochs, tests, evaluate=False)


if __name__ == "__main__":
    main([] if "__file__" not in globals() else None)
