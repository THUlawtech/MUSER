# MUSER

Source code and dataset of "MUSER: A Multi-View Similar Retrieval Dataset".

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

For total case documents `case_pool.json`, see [this link](https://cloud.tsinghua.edu.cn/f/2fcef86efc99420b8108/?dl=1).

For jieba tokenized corpus `corpus.json`, see [this link](https://cloud.tsinghua.edu.cn/f/7016e6301b654f969c3b/?dl=1).

## Keys of the Case Document

`case_pool.json` contains all cases in MUSER. The key definition of this file is as below:

```json
{
  "uid": unique id of the case,
  "caseID": case id assigned by the court,
  "content": {
    "本院查明": The court's findings of fact,
    "本院认为": The court's opinion,
    "法条": cited articles
  },
  "labels": {  # labels annotated at sentences
    "案情标签": legal fact labels
    "争议焦点": disputed focus labels
  },
  "split_labels": {  # labels splited by level
    "aqbq": legal fact labels,
    "zyjd": disputed focus labels
  }
}
```

## Experiment

For our SCR train set, see [this link](https://cloud.tsinghua.edu.cn/f/7c9231e0daaf4fd4894e/?dl=1); test set see [this link](https://cloud.tsinghua.edu.cn/f/448b8ff3202d428babe4/?dl=1).

For our fine-tuned SCR checkpoint, see [this link](https://cloud.tsinghua.edu.cn/f/ec54ceded8ab4b54ae4c/?dl=1).
