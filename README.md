# Generating LLM-Elicited TOT Queries
This repository contains scripts to generate LLM-Elicited TOT Queries. 
Synthetic TOT queries generated from this repository were used as test queries for the TREC 2024 Tip of the Tongue (TOT) track and will be used for subsequent tracks.

## 1. Python Environment
This codebase is implemented in Python 3.9.19. 
The necessary libraries can be found in `requirements.txt`. 

## 2. Data Preparation
1. The raw MS-ToT data is located at `data/ToT.txt`, which is sourced from [Tip of the Tongue Known Item Retrieval Dataset for Movie Identification](https://github.com/microsoft/Tip-of-the-Tongue-Known-Item-Retrieval-Dataset-for-Movie-Identification).
2. We utilized the `wikipediaURL` column to obtain movie names and saved the `ID`, `QuestionBody`, `wikipediaURL`, and `movieName` in the TSV file `data/ToT_processed.tsv`.
3. We found that 801 out of the 1000 movies provide a `wikipediaURL`. We removed the 199 rows with empty `movieName` and saved a new TSV file, `data/ToT_processed_filtered.tsv`.

Note: One can automatically retrieve the remaining movie names using the `imdbURL` column. Further processing is needed to match the movie names with Wikipedia page titles. In this project, we did this manually.

## 3. Eliciting TOT Queries from LLMs
Run `llm_queries/generate_gpt_queries.py` to generate LLM queries. 
The following are the core components of the script.

### 3.1 Prompt Template
We experimented with six different templates. 
The prompt templates can be found in `llm_queries/generate_gpt_queries.py`.

#### 3.1.1 Wikipedia Page Summarization
Most of the prompt templates allow for the addition of reference information about the entity. In our implementation, we append paragraphs from the Wikipedia page of the entity. Specifically for the movie domain, we use the introduction section and the Plot section if it exists. We found that most of the selected Wikipedia paragraphs can fit into the prompt without exceeding the 4096 input token limit of the GPT-4o model we used. 
You can adjust `max_paragraphs` in the `split_document` function if the context needs to be shortened.

### 3.2 Model Configuration
We use the GPT-4o model. 
`max_tokens` is set to 1024 for all experiments.

### 3.3 Query Generation
1. First, use the `generate_multiple` function to automatically generate ToT queries. A few paths in the function:
    - `input_file_path`: input the raw TSV file generated from MSToT with movie names; we use `data/ToT_processed_filtered.tsv`.
    - `output_file_path`: the output file path to save the generated queries in TSV format.
    - `json_folder`: the output folder path to save the full prompt and response for each target object to JSON files in the following format:
        ```
        result = {
            "topic": topic,
            "ToTObject": tgt_object,
            "paragraphs": paragraphs,
            "prompt": TEMPLATE_6,
            "response": response,
        }
        ```
    - `log_folder_path`: the output file path to save the log files. `process_log.txt` contains the API communication information, while `failed_objects_log.txt` contains the information about failed objects that need to be re-run in a later step.
    - `queries_file_path`: input the `queries.jsonl` based on the dataset split for which you want to generate queries. Choose from `[train, dev, test]`.

2. During generation, several checks are performed:
    - Empty paragraphs: if the paragraphs are empty, we skip the object. This occurs when no information is returned from the Wikipedia API, typically when the movie name does not match the movie name in the URL. This mainly happens with movies that share the same name but were released in different years. For example, in the MS-ToT dataset, the URL for the movie "Nowhere" is "https://en.wikipedia.org/wiki/Nowhere", but it has now become "https://en.wikipedia.org/wiki/Nowhere_(1997_film)" due to a new movie with the same name at "https://en.wikipedia.org/wiki/Nowhere_(2023_film)".
    - Max retries reached: We check if the LLM response contains the movie name. If it does, we re-run the object. We set the maximum retries to 3, and if the API call fails three times, we skip the object.
    - Other Errors: Record other errors from the API call and the processing loop.

3. After the `generate_multiple` function is finished, run the `generate_single` function to re-run the failed objects according to `failed_objects_log.txt`. You need to set the values for `topic`, `mstot_id`, `tgt_object`, and `wikipediaURL` for each re-run.

4. Finally, use `check_missing_ids` and `check_duplicate_ids_in_tsv` to ensure all objects are generated and that there are no duplicate IDs in the output TSV file.

Note: In the test split, there are two movies without a `wikipediaURL`: Fall Out (ID: 667) and Phantom Town (IDs: 884 and 931). We find movie names by IMDB ID. Note that Phantom Town exists in the dataset twice with different IDs.

## 4. Convert to TREC Style
Run `llm_queries/format_queries_in_TREC_style.py` to generate the ToT queries in TREC style. Set the paths according to the comments in the script. The output file can now be used for model training and inference, which will be utilized in the following validation step.

## 5. Validation of Generated Synthetic Queries through System Ranking Correlation

To check the validity of the generated queries, we assess the correlation between two system rankings: one generated by CQA-based queries (e.g., MS-TOT) and the other by synthetic queries.

We employ 1) lexicon-based open-source retrieval models, 2) dense open-source retrieval models, and 3) closed-source API-based models as retrieval models.

In total, we run 40 different retrieval models and rank the systems based on retrieval performance (NDCG and MRR), followed by calculating Kendall's Tau correlation between the two system rankings.

Below are the guidelines for running the retrieval models.

### 5.1 Lexicon-based Retrieval Models

Run `model_inference/run_lexicon.py` for lexicon methods. Consider the following example command for BM25:

```
python run_lexicon.py --run_id bm25_0.8_1.0_test_llm-w-para \
        --index_name bm25_0.8_1.0_test_llm_w_para \
        --index_path ./anserini_indicies \
        --docs_path ./anserini_docs \
        --data_path ./datasets-llm-w-para \
        --param_k1 0.8 --param_b 1.0 \
        --field text --query text --split test \
        --run ./results/BM25-0.8-1.0-trec-baseline/llm-w-para/test.run \
        --run_format trec_eval \
```

To use the LMDirichlet Similarity model, add `--param_mu <mu_value>` as a parameter, e.g., `--param_mu 1000`.

You can also use TF-IDF, IB Similarity, DFR Similarity, and more by changing the methods in the script.

### 5.2 Dense Models

Run `model_inference/run_dense.py` for dense models. Consider the following example command for DistilBERT from the TREC baseline:

```
python run_dense.py --freeze_base_model \
        --run_id baseline_distilbert_test_llm-w-para \
        --data_path ./datasets-llm-w-para \
        --model_or_checkpoint path/to/the/checkpoint \
        --model_dir ./results/distilbert-trec-baseline/llm-w-para \
        --embed_size 768 --encode_batch_size 256 \
        --query text --device cuda
```

You can change the model to any dense model supported in SentenceTransformers. You can also use DPR models. Remember to change `embed_size` based on the model architecture.

### 5.3 Closed-source Models
Run `llm_queries/rank_with_gpt.py` to generate rankings using GPT models. Input the TREC style queries file and the output file path to save the ranking results. The script will generate ranking results for each query. 
For all our experiments, we consistently set `temperature` to 0.5 and `max_tokens` to 1024. We follow the same prompt used in [TREC-ToT baseline](https://github.com/TREC-ToT/bench/blob/main/GPT4.md). Note that the generated ranking results come with order numbers when using the GPT-3.5 Turbo Instruct model. Apply `llm_queries/clean_gpt_ranking.py` to remove the order numbers and save the cleaned ranking results.

After generating the ranking results, you can use `model_inference/gpt_post.py` to process the ranking results and generate run files in TREC format. Consider the following example commands:

Matching the names without aliases:
```
python gpt_post.py \
    --run_id gpt_4o_llm_w_paragraph_temp-0.7-no-alias \
    --input cleaned_gpt_test_queries_llm_w_paragraph.jsonl \
    --split test \
    --data_path ./datasets-llm-w-para \
    --index_name gpt_4o_turbo_llm_w_paragraph-no-alias \
    --index_path ./anserini_indicies \
    --docs_path ./anserini_docs \
    --run gpt_4o_temp-0.7-no-alias.run \
    --run_format trec_eval \
    --run_id gpt_4o_temp-0.7-no-alias
```

Matching the names with aliases, which is expected to perform better than the above:
```
python gpt_post.py \
    --run_id gpt_4o_llm_w_paragraph_temp-0.7-with-alias \
    --input llm-w-para-temp-0.7/cleaned_gpt_test_queries_llm_w_paragraph.jsonl \
    --split test \
    --data_path ./datasets-llm-w-para \
    --index_name gpt_4o_turbo_llm_w_paragraph-with-alias \
    --index_path ./anserini_indicies \
    --docs_path ./anserini_docs \
    --run gpt_4o_temp-0.7-with-alias.run \
    --run_format trec_eval \
    --gather_wikidata_aliases \
    --wikidata_cache ./wikidata_cache/
```

### 5.4 Calculating System Rank Correlations
Run `analysis/KT_excel.ipynb` to perform Kendall's tau analysis by reading the CSV results.
