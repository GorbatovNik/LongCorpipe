import numpy as np
import pymorphy2
from collections import defaultdict, Counter
from enum import Enum
from langdetect import detect
from nltk.stem.snowball import SnowballStemmer

TAGS_ALLOWED_FOR_REPRESENTATIVES = ("PROPN", "ADJ", "CCONJ", "INTJ", "PART", "ADP")

class NormalizationType(Enum):
    CONLLU = 1
    LEMMATIZATION = 2
    STEMMING = 3


def check_if_repr(mention_info, normalization_type: NormalizationType):
    POS_tags = [tok[2] for tok in mention_info]
    if not all(t in TAGS_ALLOWED_FOR_REPRESENTATIVES for t in POS_tags):
        return None
    if "PROPN" not in POS_tags:
        return None

    normalize = lambda token, lemma: lemma.lower() # NormalizationType.CONLLU
    if normalization_type == NormalizationType.LEMMATIZATION:
        morph = pymorphy2.MorphAnalyzer()
        normalize = lambda token, lemma: morph.parse(token.lower())[0].normal_form
    elif normalization_type == NormalizationType.STEMMING:
        stemmer_en = SnowballStemmer("english")
        stemmer_ru = SnowballStemmer("russian")

        def stem_token(token):
            try:
                lang = detect(token)
            except:
                lang = 'ru'
            if lang == 'ru':
                return stemmer_ru.stem(token.lower())
            else:
                return stemmer_en.stem(token.lower())

        normalize = lambda token, lemma: stem_token(token)

    return tuple(sorted([
        normalize(token, lemma)
        for token, lemma, POS_tag in mention_info
        if POS_tag == "PROPN"  # representative is formed only from PROPN
    ]))


def get_repr(cluster_mentions):
    results = []
    for mention in cluster_mentions:
        lemmas = check_if_repr(mention.info, NormalizationType.CONLLU)
        if lemmas:
            results.append(lemmas)

    if not results:
        return None

    return Counter(results)


# def merge_clusters_by_representative(clusters):
#     merged = []
#     representatives = []

#     for cluster in clusters:
#         cur_repr = get_repr(cluster)
#         if cur_repr is None:
#             merged.append(cluster)
#             representatives.append(cur_repr)
#             continue

#         for i, repr in enumerate(representatives):
#             if cur_repr == repr:
#                 merged[i].extend(cluster)
#                 break
#         else:
#             merged.append(cluster)
#             representatives.append(cur_repr)

#     return merged


def agario(clusters):
    merged_clusters = []
    representatives = []
    top_candidates = []

    for cluster in clusters:
        cur_counter = get_repr(cluster)

        if cur_counter is None:
            merged_clusters.append(cluster)
            representatives.append(None)
            top_candidates.append(None)
            continue

        cur_top = cur_counter.most_common(1)[0][0]
        merged = False
        i = 0

        while i < len(merged_clusters):
            rep_counter = representatives[i]
            rep_top = top_candidates[i]

            if rep_counter is None:
                i += 1
                continue

            if len(cur_counter) <= len(rep_counter):
                smaller, larger = cur_counter, rep_counter
                top_candidate_larger = rep_top
            else:
                smaller, larger = rep_counter, cur_counter
                top_candidate_larger = cur_top

            if top_candidate_larger in set(smaller):
                if len(cur_counter) <= len(rep_counter):
                    merged_clusters[i].extend(cluster)
                    representatives[i] += cur_counter
                    top_candidates[i] = representatives[i].most_common(1)[0][0]
                    merged = True
                    break
                else:
                    cluster.extend(merged_clusters[i])
                    cur_counter += rep_counter
                    cur_top = cur_counter.most_common(1)[0][0]
                    del merged_clusters[i]
                    del representatives[i]
                    del top_candidates[i]

                    continue

            i += 1

        if not merged:
            merged_clusters.append(cluster)
            representatives.append(cur_counter)
            top_candidates.append(cur_top)

    return merged_clusters


def merge_clusters(
        subdocs: list[list[tuple]]
):
    """
    Args:
    subdocs (list[list[Mention]]):
        [
            [
                Mention(
                    sent_id: int
                    begin: int  # first token index in sentence
                    end: int  # last token index in sentence (inclusive)
                    info: list[Tuple[token: str, lemma :str, POS_tag: str]]  # token properties
                    cluster: int  # cluster id
                ),
                ...
            ],
            ...
        ]
    Returns list[list[Mention]]:
        [
            [ # cluster
                Mention(...),
                ...
            ],
            ...
        ]
    """
    clusters = defaultdict(list)
    for subdoc in subdocs:
        for mention in subdoc:
            clusters[mention.cluster].append(mention)
    final_clusters = list(clusters.values())

    # representive mentions merging
    if len(final_clusters) > 1:
        final_clusters = agario(final_clusters)

    return final_clusters
