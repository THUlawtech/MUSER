# MUSER

Source code and dataset for "MUSER: A Multi-View Similar Retrieval Dataset".

## Dataset Structure

`/MUSER/data` is the root directory of all MUSER data. The data structure is shown below:

```
data
- cases
  - cands_by_querys.json  # queries and corresponding candidates IDs
  - train_test.json  # train set and test set for deep nerual model training
- labels
  - golden_labels.json  # queries and relevant candidates IDs
  - sim_scores.json  # relevance scores between queies and candidates
  - top30_dict.json  # top 30 relevent candidates
- legal_items  # legal element label schema
  - aqbq.json  # legal fact
  - ft.json  # law statutory
  - zyjd.json  # dispute focus
- predictions  # retrieval results of baseline models
  - bm25_top100.json
  - labels_top100.json
  - lfm_top100.json
  - lmir_top100.json
  - tfidf_top100.json
- utils
  - stopword.txt
```

For total case documents `case_pool.json`, see [this link](https://drive.google.com/file/d/15TgaACV8hilVeMl8zUTUuZ4RKqNPQ_38/view?usp=sharing).

For jieba tokenized corpus `corpus.json`, see [this link](https://drive.google.com/file/d/1bxayhEsU6NvTVoWggy24NFT4tSacf68w/view?usp=sharing).

## Legal Element Label Schema

For our hierarchical legal element label schema and their English meanings, see [this link](https://docs.google.com/spreadsheets/d/1VTLhiPLrCufI_Zhd4bjcp7zlqC7mJEkp/edit?usp=sharing&ouid=103483335604509478099&rtpof=true&sd=true).

## Keys of the Case Document

`case_pool.json` contains all cases in MUSER. The key definition of this file is as below:

```json
{
  "uid": "unique id of the case",
  "caseID": "case id assigned by the court",
  "content": {
    "本院查明": "The court's findings of fact",
    "本院认为": "The court's opinion",
    "法条": "cited articles"
  },
  "labels": {
    "案情标签": "legal fact labels annotated at sentences",
    "争议焦点": "disputed focus labels annotated at sentences"
  },
  "split_labels": {
    "aqbq": "legal fact labels splited by level",
    "zyjd": "disputed focus labels splited by level"
  }
}
```

## Experiment

### Legal Element Prediction

Run `/src/experiments/label_pred/data_preprocess.py` to preprocess the sentence-level training data.

For legal fact label prediction, run `/src/experiments/label_pred/train.py`. For dispute focus prediction, run `/src/experiments/label_pred/train_zhengyi.py`. There are two training models you can select: BERT and Lawformer.

### Similar Case Retrieval

For traditional bag-of-word retrieval models (BM25, TF-IDF, and LMIR) and our proposed legal-element-based model, run `/src/experiments/solve_{model_name}.py` to get retrieval results at `/data/predictions/{model_name}_top100.json`.

For deep neural model based on Lawformer, run `/src/experiments/train.sh` to fine-tune it, run `/src/experiments/test.py` to get the retrieval result `lfm_top100.json`.

For evaluation, run `/src/experiments/metrics.py`, results will be printed on the console.

For our deep neural model train set, see [this link](https://drive.google.com/file/d/1_q2WgfDH84aAbWfytZz4j6cJEW-CznJl/view?usp=sharing); test set see [this link](https://drive.google.com/file/d/1xdHAATu1_5yNRiOMjnBfJHpGVzhTScpL/view?usp=sharing).

For our fine-tuned checkpoint, see [this link](https://drive.google.com/file/d/1XGPhE4Cwr6mXzQ1zJb856ofJ9LeMgp0m/view?usp=sharing).
