
event_columns = ['oid', 'verbs', 'skeleton_words_clean',
                 'skeleton_words', 'words', 'pattern', 'frequency']
relation_senses = [
        'Precedence', 'Succession', 'Synchronous',
        'Reason', 'Result',
        'Condition', 'Contrast', 'Concession',
        'Conjunction', 'Instantiation', 'Restatement', 'ChosenAlternative', 'Alternative', 'Exception',
        'Co_Occurrence']

relation_table_name = "Relations"
relation_columns = ['aser_id', 'event1_id', 'event2_id'] + relation_senses

USE_EXPANSION = False
USE_SELF_REL = False

db_path = "../data/database/core/KG_v0.1.0.db"
label_path = "./data/labels.json"
query_path = "./data/label2queries.json"
matched_event_path = "./data/label2aser_ids.json"
related_relations_path = "./data/related_relations.tsv"
selected_aser_info_path = "./data/selected_aserid2info.json"
extracted_relations_path = "./data/extracted_label_rels.json"