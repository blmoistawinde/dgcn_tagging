import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm_notebook
from collections import defaultdict, Counter
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, GATConv
import pickle
from scipy.sparse import csr_matrix


FINE_LABELS = ['1-1_small-sounding-engine',
               '1-2_medium-sounding-engine',
               '1-3_large-sounding-engine',
               '1-X_engine-of-uncertain-size',
               '2-1_rock-drill',
               '2-2_jackhammer',
               '2-3_hoe-ram',
               '2-4_pile-driver',
               '2-X_other-unknown-impact-machinery',
               '3-1_non-machinery-impact',
               '4-1_chainsaw',
               '4-2_small-medium-rotating-saw',
               '4-3_large-rotating-saw',
               '4-X_other-unknown-powered-saw',
               '5-1_car-horn',
               '5-2_car-alarm',
               '5-3_siren',
               '5-4_reverse-beeper',
               '5-X_other-unknown-alert-signal',
               '6-1_stationary-music',
               '6-2_mobile-music',
               '6-3_ice-cream-truck',
               '6-X_music-from-uncertain-source',
               '7-1_person-or-small-group-talking',
               '7-2_person-or-small-group-shouting',
               '7-3_large-crowd',
               '7-4_amplified-speech',
               '7-X_other-unknown-human-voice',
               '8-1_dog-barking-whining']

WHITE_LIST = ['2-1_rock-drill',
            '2-2_jackhammer',
            '2-3_hoe-ram',
            '2-4_pile-driver',
            '4-1_chainsaw',
            '5-1_car-horn',
            '5-2_car-alarm',
            '5-3_siren',
            '5-4_reverse-beeper',
            '6-1_stationary-music',
            '6-2_mobile-music',
            '6-3_ice-cream-truck',
            '7-3_large-crowd',
            '7-4_amplified-speech',
            '8-1_dog-barking-whining']

COARSE_LABELS = ['1_engine',
                 '2_machinery-impact',
                 '3_non-machinery-impact',
                 '4_powered-saw',
                 '5_alert-signal',
                 '6_music',
                 '7_human-voice',
                 '8_dog']

base_dir = "./data/"

ALL_LABELS = COARSE_LABELS + FINE_LABELS

with open(base_dir+"extracted_label_rels.json", encoding="utf-8") as f:
    extracted_rels = json.load(f)

# ontology graph
coarse_indices = np.arange(len(COARSE_LABELS))
fine_indices = np.arange(len(FINE_LABELS)) + len(COARSE_LABELS)
N = len(COARSE_LABELS) + len(FINE_LABELS)
adj_ontology = np.zeros((N, N), dtype=int)
for cid, clabel in enumerate(COARSE_LABELS):
    for fid, flabel in enumerate(FINE_LABELS):
        if int(flabel[0]) == int(clabel[0]):
            adj_ontology[coarse_indices[cid], fine_indices[fid]] = 1
            adj_ontology[fine_indices[fid], coarse_indices[cid]] = 1
dgl_ontology = dgl.from_scipy(csr_matrix(adj_ontology))
dgl_ontology = dgl_ontology.add_self_loop()
print(dgl_ontology)

with open(base_dir+"ontology_cid.pkl", "wb") as f:
    pickle.dump([dgl_ontology, coarse_indices], f)
with open(base_dir+"ontology_fid.pkl", "wb") as f:
    pickle.dump([dgl_ontology, fine_indices], f)

# aser graph with heuristic filtering
adj_aser_pre_conj_wt = np.zeros((N, N), dtype=int)
for rels in extracted_rels.values():
    if not ("Precedence" in rels or "Conjunction" in rels):
        continue
    id1 = ALL_LABELS.index(rels["name1"])
    id2 = ALL_LABELS.index(rels["name2"])
    # only allow whitelist fine labels
    if id1 >= len(COARSE_LABELS) and rels["name1"] not in WHITE_LIST:
        continue
    if id2 >= len(COARSE_LABELS) and rels["name2"] not in WHITE_LIST:
        continue
    adj_aser_pre_conj_wt[id1, id2] = 1
    adj_aser_pre_conj_wt[id2, id1] = 1

dgl_aser_pre_conj_wt0 = dgl.from_scipy(csr_matrix(adj_aser_pre_conj_wt))
dgl_aser_pre_conj_wt = dgl_aser_pre_conj_wt0.add_self_loop()
print(dgl_aser_pre_conj_wt)

with open(base_dir+"aser_pre_conj_wt_cid.pkl", "wb") as f:
    pickle.dump([dgl_aser_pre_conj_wt, coarse_indices], f)
with open(base_dir+"aser_pre_conj_wt_fid.pkl", "wb") as f:
    pickle.dump([dgl_aser_pre_conj_wt, fine_indices], f)

# combine them in one graph (ASER+OT)
adj_ontology_aser_pre_conj_wt = np.maximum(adj_ontology, adj_aser_pre_conj_wt)
dgl_ontology_aser_pre_conj_wt = dgl.from_scipy(csr_matrix(adj_ontology_aser_pre_conj_wt))
dgl_ontology_aser_pre_conj_wt = dgl_ontology_aser_pre_conj_wt.add_self_loop()
print(dgl_ontology_aser_pre_conj_wt)


with open(base_dir+"ontology_aser_pre_conj_wt_cid.pkl", "wb") as f:
    pickle.dump([dgl_ontology_aser_pre_conj_wt, coarse_indices], f)
with open(base_dir+"ontology_aser_pre_conj_wt_fid.pkl", "wb") as f:
    pickle.dump([dgl_ontology_aser_pre_conj_wt, fine_indices], f)

