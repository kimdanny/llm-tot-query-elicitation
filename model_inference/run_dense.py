import ir_datasets
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from src import data, encode, utils
import pytrec_eval
from torch import nn
import argparse
import os
import logging

import tot
import run_lexicon

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl
import torch.nn.init as init

log = logging.getLogger(__name__)


class CustomDPRModel:
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def encode(
        self,
        texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=False,
        device="cuda",
    ):
        all_embeddings = []
        iterable = tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding",
            disable=not show_progress_bar,
        )
        for i in iterable:
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs).pooler_output
            if convert_to_numpy:
                outputs = outputs.cpu().numpy()
            all_embeddings.append(outputs)
        if convert_to_numpy:
            return np.concatenate(all_embeddings, axis=0)
        return torch.cat(all_embeddings, dim=0)

    def eval(self):
        self.model.eval()


class LouisDPRModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(1024, 1024)
        self.tanh = nn.Tanh()

    def encode(self, inputs, attention_mask):
        model_output = self.lm(inputs, attention_mask)
        token_embeddings = model_output[0]
        cls_embeddings = self.cls_pooling(token_embeddings)
        return cls_embeddings

    def mean_pooling(self, x, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        return torch.sum(x * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def cls_pooling(self, x):
        return x[:, 0]

    def forward(self, input_ids, attention_mask=None):
        cls_embeddings = self.encode(input_ids, attention_mask)
        projected = self.projection(cls_embeddings)
        activated = self.tanh(projected)
        return activated

    def load_model(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Remove the 'lm.' prefix and load
            new_state_dict = {
                key.replace("lm.", ""): value for key, value in state_dict.items()
            }
            # Separate state dict for the pre-trained and custom layers
            lm_state_dict = {
                key: val
                for key, val in new_state_dict.items()
                if not key.startswith("projection")
            }
            projection_state_dict = {
                key: val
                for key, val in new_state_dict.items()
                if key.startswith("projection")
            }

            self.lm.load_state_dict(lm_state_dict, strict=False)  # Load the model part
            self.projection.load_state_dict(
                projection_state_dict, strict=False
            )  # Load the projection part
            print("Model loaded successfully.")
        except RuntimeError as e:
            print("Failed to load model state dictionary!")
            print(e)

    def eval_model(self):
        self.eval()


def count_parameters(model, trainable=True):
    """Returns the total number of parameters, optionally filtering only trainable parameters."""
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "train_dense", description="Trains a dense retrieval model"
    )

    parser.add_argument(
        "--data_path", default="./datasets/TREC-TOT", help="location to dataset"
    )

    parser.add_argument(
        "--negatives_path",
        default="./bm25_negatives",
        help="path to folder containing negatives ",
    )

    parser.add_argument(
        "--query", choices=["title", "text", "title_text"], default="title_text"
    )

    parser.add_argument(
        "--model_or_checkpoint",
        type=str,
        required=True,
        help="hf checkpoint/ path to pt-model",
    )
    parser.add_argument(
        "--embed_size", required=True, type=int, help="hidden size of the model"
    )
    parser.add_argument(
        "--epochs", type=int, default=0, help="number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size (training)"
    )
    parser.add_argument(
        "--encode_batch_size", type=int, default=124, help="batch size (inference)"
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=-1,
        help="steps before evaluation is run",
    )

    parser.add_argument(
        "--freeze_base_model",
        action="store_true",
        default=False,
        help="if set, freezes the base layer and trains only a projection layer on top",
    )
    parser.add_argument(
        "--metrics",
        required=False,
        default=run_lexicon.METRICS,
        help="csv - metrics to evaluate",
    )
    parser.add_argument(
        "--n_hits", default=1000, type=int, help="number of hits to retrieve"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="device to train /evaluate model on"
    )

    parser.add_argument(
        "--model_dir", type=str, help="folder to store model & runs", required=True
    )
    parser.add_argument(
        "--run_id", required=True, help="run id (required if run_format = trec_eval)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negatives_out",
        default=None,
        help="if provided, dumps negatives for use in training other models",
    )
    parser.add_argument(
        "--n_negatives", default=10, type=int, help="number of negatives to obtain"
    )
    parser.add_argument(
        "--corrupt_method", default=None, help="methods to re-initialize model layers"
    )

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    utils.set_seed(args.seed)
    log.info(f"args: {args}")

    tot.register(args.data_path)
    metrics = args.metrics.split(",")

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    if args.freeze_base_model:
        if args.model_or_checkpoint == "facebook/dpr-question_encoder-single-nq-base":
            model = CustomDPRModel(args.model_or_checkpoint, device=args.device)
        elif args.model_or_checkpoint == "YOUR OWN MODEL NAME":
            model = LouisDPRModel("OpenMatch/co-condenser-large-msmarco")
            model = model.to(args.device)
            model.load_model(args.model_or_checkpoint)
        elif args.model_or_checkpoint == "dense_models/baseline_distilbert_ckpt/model":
            model = SentenceTransformer(args.model_or_checkpoint, device=args.device)
            transformer_model = model._modules["0"].auto_model

            # the second last transformer layer
            second_last_layer = transformer_model.transformer.layer[4]

            if args.corrupt_method == "kaiming_normal":
                # Re-initialize weights of the attention
                init.kaiming_normal_(
                    second_last_layer.attention.q_lin.weight, nonlinearity="relu"
                )
                init.kaiming_normal_(
                    second_last_layer.attention.k_lin.weight, nonlinearity="relu"
                )
                init.kaiming_normal_(
                    second_last_layer.attention.v_lin.weight, nonlinearity="relu"
                )
                init.kaiming_normal_(
                    second_last_layer.attention.out_lin.weight, nonlinearity="relu"
                )
            elif args.corrupt_method == "xavier_normal":
                init.xavier_normal_(second_last_layer.attention.q_lin.weight)
                init.xavier_normal_(second_last_layer.attention.k_lin.weight)
                init.xavier_normal_(second_last_layer.attention.v_lin.weight)
                init.xavier_normal_(second_last_layer.attention.out_lin.weight)
            else:
                print("Passed None or invalid corrupt methods!")

            log.info(
                f"the second last transformer layer corrupted with {args.corrupt_method}"
            )

            # re-initialize biases to zero
            second_last_layer.attention.q_lin.bias.data.zero_()
            second_last_layer.attention.k_lin.bias.data.zero_()
            second_last_layer.attention.v_lin.bias.data.zero_()
            second_last_layer.attention.out_lin.bias.data.zero_()

        else:
            base_model = SentenceTransformer(
                args.model_or_checkpoint, device=args.device, trust_remote_code=True
            )
            for param in base_model.parameters():
                param.requires_grad = False
            projection = models.Dense(
                args.embed_size, args.embed_size, activation_function=nn.Tanh()
            )
            model = SentenceTransformer(
                modules=[base_model, projection], device=args.device
            )
    else:
        model = SentenceTransformer(args.model_or_checkpoint, device=args.device)

    # print("Total trainable parameters:", count_parameters(model))
    # print("Total parameters (including non-trainable):", count_parameters(model, trainable=False))

    irds_splits = {}
    st_data = {}

    # splits
    for split in {"train", "dev"}:
        irds_splits[split] = ir_datasets.load(f"trec-tot:{split}")

        log.info(f"loaded split {split}")
        st_data[split] = data.SBERTDataset(
            irds_splits[split],
            query_type=args.query,
            negatives=utils.read_json(
                os.path.join(
                    args.negatives_path, f"{split}-{args.query}-negatives.json"
                )
            ),
        )

    # log.info(f"training model for {args.epochs} epochs")
    # train_dataloader = DataLoader(st_data["train"], shuffle=True, batch_size=args.batch_size)

    # args.loss_fn = "mnrl"
    # if args.loss_fn == "mnrl":
    #     train_loss = losses.MultipleNegativesRankingLoss(model=model)
    # else:
    #     raise NotImplementedError(args.loss_fn)

    val_evaluator = data.get_ir_evaluator(
        st_data["dev"],
        name=f"dev",
        mrr_at_k=[1000],
        ndcg_at_k=[10, 1000],
        corpus_chunk_size=args.encode_batch_size,
    )

    # optimizer_params = {
    #     "lr": args.lr
    # }

    # Tune the model
    # model.fit(train_objectives=[(train_dataloader, train_loss)],
    #           evaluation_steps=args.evaluation_steps,
    #           output_path=os.path.join(model_dir, "model"),
    #           evaluator=val_evaluator,
    #           epochs=args.epochs,
    #           warmup_steps=args.warmup_steps,
    #           optimizer_params=optimizer_params,
    #           weight_decay=args.weight_decay,
    #           save_best_model=True)

    log.info("encoding corpus with model")
    embed_size = args.embed_size
    index, (idx_to_docid, docid_to_idx) = encode.encode_dataset_faiss(
        model,
        embedding_size=embed_size,
        dataset=irds_splits["train"],
        device=args.device,
        encode_batch_size=args.encode_batch_size,
        model_name=args.model_or_checkpoint,
    )

    runs = {}
    eval_res_agg = {}
    eval_res = {}

    try:
        log.info("attempting to load test set")
        # plug in the test set
        irds_splits["test"] = ir_datasets.load(f"trec-tot:test")
        log.info("success!")
    except KeyError:
        log.info("couldn't find test set!")
        pass

    split_qrels = {}
    for split, dataset in irds_splits.items():
        log.info(f"running & evaluating {split}")

        run = encode.create_run_faiss(
            model=model,
            dataset=dataset,
            query_type=args.query,
            device=args.device,
            eval_batch_size=args.encode_batch_size,
            index=index,
            idx_to_docid=idx_to_docid,
            docid_to_idx=docid_to_idx,
            top_k=args.n_hits,
            model_name=args.model_or_checkpoint,
        )
        runs[split] = run

        if dataset.has_qrels():
            qrel, n_missing = utils.get_qrel(dataset, run)
            split_qrels[split] = qrel
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)

            eval_res[split] = evaluator.evaluate(run)
            eval_res_agg[split] = utils.aggregate_pytrec(eval_res[split], "mean")

            for metric, (mean, std) in eval_res_agg[split].items():
                log.info(f"{metric:<12}: {mean:.4f} ({std:0.4f})")

    utils.write_json(
        {
            "aggregated_result": eval_res_agg,
            "run": runs,
            "result": eval_res,
            "args": vars(args),
        },
        os.path.join(model_dir, "out.gz"),
        zipped=True,
    )

    run_id = args.run_id
    assert run_id is not None

    for split, run in runs.items():
        run_path = os.path.join(model_dir, f"{split}.run")
        with open(run_path, "w") as writer:
            for qid, r in run.items():
                for rank, (doc_id, score) in enumerate(
                    sorted(r.items(), key=lambda _: -_[1])
                ):
                    writer.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_id}\n")

    if args.negatives_out:
        log.info(f"writing negatives to folder: {args.negatives_out}")
        os.makedirs(args.negatives_out, exist_ok=True)
        out = {}

        for split, run in runs.items():
            if split == "test":
                continue
            negatives_path = os.path.join(
                args.negatives_out, f"{split}-{args.query}-negatives.json"
            )
            qrel = split_qrels[split]
            for qid, hits in run.items():
                hits = sorted(hits.items(), key=lambda _: -_[1])
                negs = []
                for (doc, score) in hits:
                    if qrel[qid].get(doc, 0) > 0:
                        continue
                    if len(negs) == args.n_negatives:
                        break
                    negs.append(doc)
                out[qid] = negs
            utils.write_json(out, negatives_path)
