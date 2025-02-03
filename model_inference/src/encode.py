import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import AutoTokenizer

import ir_datasets

from src import data, utils
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


def encode_dataset_faiss(
    model: SentenceTransformer,
    embedding_size: int,
    dataset: ir_datasets.Dataset,
    device,
    encode_batch_size,
    model_name,
):
    doc_ids, documents = data.get_documents(dataset)

    idx_to_docid = {}
    docid_to_idx = {}
    for idx, doc_id in enumerate(doc_ids):
        idx_to_docid[idx] = doc_id
        docid_to_idx[doc_id] = idx

    if model_name == "YOUR OWN MODEL NAME":
        model.eval_model()
        tokenizer = AutoTokenizer.from_pretrained(
            "OpenMatch/co-condenser-large-msmarco"
        )
        all_embeddings = []
        iterable = tqdm(range(0, len(documents), encode_batch_size), desc="Encoding")
        # for i in range(0, len(documents), encode_batch_size):
        #     batch_docs = documents[i:i+encode_batch_size]
        #     inputs = tokenizer(batch_docs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        #     inputs = {k: v.to(device) for k, v in inputs.items()}
        #     with torch.no_grad():
        #         outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], show_progress_bar=True)
        #     all_embeddings.extend(outputs.cpu().numpy())
        for i in iterable:
            batch_docs = documents[i : i + encode_batch_size]
            inputs = tokenizer(
                batch_docs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(
                    inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
            all_embeddings.extend(outputs.cpu().numpy())

        embeddings = np.array(all_embeddings)
    else:
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(
                documents,
                batch_size=encode_batch_size,
                show_progress_bar=True,
                device=device,
                convert_to_numpy=True,
            )

    index = faiss.IndexFlatIP(embedding_size)
    indexwmap = faiss.IndexIDMap(index)
    indexwmap.add_with_ids(embeddings, np.arange(len(doc_ids)))

    return indexwmap, (idx_to_docid, docid_to_idx)


def create_run_faiss(
    model: SentenceTransformer,
    dataset: ir_datasets.Dataset,
    query_type,
    device,
    eval_batch_size,
    index: faiss.IndexIDMap,
    idx_to_docid,
    docid_to_idx,
    top_k,
    model_name,
):

    qids = []
    queries = []
    for query in dataset.queries_iter():
        queries.append(utils.get_query(query, query_type))
        qids.append(query.query_id)

    if model_name == "YOUR OWN MODEL NAME":
        model.eval_model()
        tokenizer = AutoTokenizer.from_pretrained(
            "OpenMatch/co-condenser-large-msmarco"
        )
        inputs = tokenizer(
            queries, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(
                inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
        query_embeddings = outputs.cpu().numpy()
    else:
        model.eval()
        with torch.no_grad():
            query_embeddings = model.encode(
                queries,
                batch_size=eval_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=device,
            )

    scores, raw_doc_ids = index.search(query_embeddings, k=top_k)
    run = {}
    for qid, sc, rdoc_ids in zip(qids, scores, raw_doc_ids):
        run[qid] = {}
        for s, rdid in zip(sc, rdoc_ids):
            if rdid == -1:
                log.warning(f"invalid doc ids!")
                continue
            run[qid][idx_to_docid[rdid]] = float(s)

    return run


def create_qrel(dataset, run=None):
    qrel = {}
    n_missing = 0
    for q in dataset.qrels_iter():
        if run and q.query_id not in run:
            n_missing += 1
        qrel[q.query_id] = {q.doc_id: q.relevance}

    return qrel, n_missing
