from constants import *
import json
import re
from tqdm import tqdm
if USE_EXPANSION:
    from nltk.corpus import wordnet as wn 
    from nltk.wsd import lesk
    from pywsd.lesk import cosine_lesk, simple_lesk
import spacy
from itertools import product
from copy import deepcopy
from nltk.stem import WordNetLemmatizer
import sqlite3
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json

from collections import defaultdict, Counter
from tqdm import trange, tqdm
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch([{'host':'localhost','port':8670}])

conn = sqlite3.connect(db_path)
nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])
lemmatizer = WordNetLemmatizer()

hypo = lambda s: s.hyponyms()
hyper = lambda s: s.hypernyms()

def expand_term(term, des, related_words = [], option = 'lesk', use_hypo = False):
    term = re.sub(' ', '_', term)
    if len(related_words) > 0:
        des += " " + " ".join(related_words)
    # lesk
    wsd_term = None
    if(option == 'lesk'):
        wsd_term = simple_lesk(des, term)   
    if(wsd_term):
        if(not use_hypo):
            all_lemmas = set(wsd_term.lemmas())
        else:
            all_lemmas = set(wsd_term.lemmas())
            for synset in list((wsd_term).closure(hypo)):
                all_lemmas |= set(synset.lemmas())
        ret = set()
        for lemma in all_lemmas:
            pos0 = lemma.synset().pos()
            ret.add(lemmatizer.lemmatize(lemma.name(), pos=pos0))
            try:
                for form in lemma.derivationally_related_forms():
                    pos0 = form.synset().pos()
                    if pos0 != "n":
                        ret.add(lemmatizer.lemmatize(form.name(), pos=pos0))
            except:
                continue
        return [t.replace("_", " ") for t in ret]
    else:
        return [term.replace("_", " ")]

# specific to the characteristic of labels
def preprocess_label(label):
    label = label[label.find("_")+1:].replace("-", " ")
    label = " ".join(x.lemma_ for x in nlp(label))
    return label

def make_dsl_text(query):
    dsl = {
        'query': {
            'multi_match': {
                'query': query,
                'fields': ["verb^2", "words", 'skeleton_words']
            }
        }
    }
    return dsl

def make_dsl_freq_weighted(query):
    dsl = {
        'query': {
            "function_score": {
                "query": {
                    'multi_match': {
                        'query': query,
                        'fields': ["verb^2", "words", 'skeleton_words']
                    }
                },
                "functions": [
                    {
                        "script_score": { 
                            "script": "_score + Math.log1p(doc['frequency'].value)",
                        }
                    }
                ]
            }
        }
    }
    return dsl

def query_event(eid):
    ret = conn.execute(f"SELECT * FROM Eventualities WHERE _id='{eid}'")
    return dict(zip(event_columns, next(ret)))

if __name__ == "__main__":
    # make queries
    labels = json.load(open(label_path, encoding="utf-8"))
    print(len(labels))
    label2queries = {}
    for label in labels:
        label_pro = preprocess_label(label)
        if USE_EXPANSION:
            # when description is not availble
            queries = list(set(expand_term(label_pro, des="")))
        else:
            queries = [label_pro]
        label2queries[label] = queries

    with open(query_path, "w", encoding="utf-8", newline='\n') as f:
        json.dump(label2queries, f, ensure_ascii=False, indent=4)

    # search es to get aser ids
    label2aser_ids = defaultdict(list)
    for label, queries in label2queries.items():
        for query in queries:
            for make_dsl in [make_dsl_text, make_dsl_freq_weighted]:
                dsl = make_dsl(query)
                result = es.search(index='aser', body=dsl)
                for hit in result['hits']['hits']:
                    aser_id = hit["_source"]["oid"]
                    if hit["_source"]["frequency"] >= 5:
                        label2aser_ids[label].append(aser_id)
        label2aser_ids[label] = list(set(label2aser_ids[label]))

    with open(matched_event_path, "w", encoding="utf-8", newline='\n') as f:
        json.dump(label2aser_ids, f)

    # use aser_id 2 retrieve info
    all_related_relations = []
    aser_ids = set(aser_id for aser_ids in label2aser_ids.values() for aser_id in aser_ids)
    ids_string = ', '.join("'"+id0+"'" for id0 in aser_ids)
    select_table = f"SELECT * FROM Relations WHERE event1_id IN ({ids_string})"
    for x in conn.execute(select_table):
        all_related_relations.append(x)
    select_table = f"SELECT * FROM Relations WHERE event2_id IN ({ids_string})"
    for x in conn.execute(select_table):
        all_related_relations.append(x)
    
    df_related_relations = pd.DataFrame(all_related_relations, columns=relation_columns)
    df_related_relations = df_related_relations.set_index("aser_id")
    df_related_relations = df_related_relations.drop_duplicates()
    print(len(df_related_relations))
    # print(df_related_relations.describe())
    df_related_relations.to_csv(related_relations_path, sep='\t')

    # save used events for EDA
    select_table = f"SELECT * FROM Eventualities WHERE _id IN ({ids_string});"
    selected_aserid2info = {}
    for x in conn.execute(select_table):
        info0 = dict(zip(event_columns, x))
        id0 = info0["oid"]
        selected_aserid2info[id0] = info0
    print(len(selected_aserid2info))
    with open(selected_aser_info_path, "w", encoding="utf-8") as f:
        json.dump(selected_aserid2info, f, ensure_ascii=False, indent=4)

    # extract relations
    aser_id2labels = defaultdict(list)
    for label, aser_ids in label2aser_ids.items():
        for aser_id in aser_ids:
            aser_id2labels[aser_id].append(label)
    # print(len(aser_id2labels))

    extracted_label_rels = {}
    for rid, row in df_related_relations.iterrows():
        eid1, eid2 = row["event1_id"], row["event2_id"]
        if eid1 in aser_id2labels and eid2 in aser_id2labels:
            rel_cnts = {}
            for k_rel, v_rel in row.items():
                if k_rel == "event1_id" or k_rel == "event2_id":
                    continue
                if v_rel > 0:
                    rel_cnts[k_rel] = v_rel
            for label1 in aser_id2labels[eid1]:
                for label2 in aser_id2labels[eid2]:
                    if not USE_SELF_REL and label1 == label2:
                        continue
                    if (label1, label2) not in extracted_label_rels:
                        extracted_label_rels[(label1, label2)] = {"aser_clues": [(eid1, eid2)]}
                        extracted_label_rels[(label1, label2)]["name1"] = label1
                        extracted_label_rels[(label1, label2)]["name2"] = label2
                    else:
                        if (eid1, eid2) in extracted_label_rels[(label1, label2)]["aser_clues"]:
                            continue     # not using duplicate ASER relations
                        extracted_label_rels[(label1, label2)]["aser_clues"].append((eid1, eid2))
                    for k_rel, v_rel in rel_cnts.items():
                        if k_rel in extracted_label_rels[(label1, label2)]:
                            extracted_label_rels[(label1, label2)][k_rel] += v_rel
                        else:
                            extracted_label_rels[(label1, label2)][k_rel] = v_rel

    extracted_label_rels2 = {f"{k1},{k2}":v for (k1, k2), v in extracted_label_rels.items()}
    print(len(extracted_label_rels2))
    with open(extracted_relations_path, "w", encoding="utf-8") as f:
        json.dump(extracted_label_rels2, f, ensure_ascii=False, indent=4)