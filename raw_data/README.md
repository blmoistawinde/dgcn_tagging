# Ontology links data for enrichment

Please refer to the original websites for [AudioSet](https://research.google.com/audioset/download.html), [SONYC](https://zenodo.org/record/3233082) and [ASER](https://hkust-knowcomp.github.io/ASER/html/index.html) (we used ASER v1 in our experiments, while they have updated to v2, which might introduce differences). 

Here we provide the processed data and the code for the processing (the resulting graph data is already in the directories of code), detailed below:

- sonyc
    - constants.py: specify the constants and directories of the i/o data
    - pipeline.py: the pipeline to mine relations for SONYC events from ASER
        1. term to queries with potential wordnet expansion
        2. use query to search for ASER events using ElasticSearch (inserted into ES from SQL beforehand)
        3. query in ASER SQL database for relations
        4. attach ASER events and relations to the original ontology
    - build_sonyc_graph.py: produce the graph structure of the OT/ASER/OT+ASER graph with the extracted relations
    - data/: input and output files
- audioset
    - build_dgl_graphs.ipynb: roduce the graph structure of the OT/ASER/OT+ASER graph with the extracted relations
    - ontology.json: the original AudioSet ontology
    - adjusted_ontology_exp: the adjusted AudioSet ontology along with the expanded queries
    - audio_id2aser_id_freqs_sel.json: a json with audioset id as key and list of (aser_id, frequency in ASER) as value. The frequency is used to filter out infrequent events and weigh the ES search.
    - extracted_audioset_rels_sel.json
      - The enhanced audioset ontology with the corresponing ASER clues, the format is like

```json
"/m/01j423-/m/09x0r": {
    "aser_clues": [
        [
            "d8c43751d4639f2d49106b3232beb25c8181e4ad",
            "b273936fe507b1c7a45a770554ad5d8528617a79"
        ],
        [
            "d8c43751d4639f2d49106b3232beb25c8181e4ad",
            "e826249bc69b91297e638d6e097dae1d670ad195"
        ]
    ],
    "name1": "Yawn",
    "name2": "Speech",
    "Precedence": 2.0,
    "Co_Occurrence": 4.0
}
```
where the key is a pair of AudioSet event id, `aser_clues` is the list of ASER event-pairs which supports the enhanced link between the AudioSet events. `name1` and `name2` is the name of the AudioSet events, the remaining keys like `Co_Occurrence` are ASER relation types and the cumulative strength according to all `aser_clues`.