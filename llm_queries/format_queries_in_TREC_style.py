import csv
import json


def load_llm_data(tsv_path):
    """Load data from raw TSV into a dictionary for quick access."""
    llm_data = {}
    with open(tsv_path, encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")
        for row in reader:
            llm_data[row["ID"]] = row["QuestionBody"]
    return llm_data


def process_queries(jsonl_input_path, jsonl_output_path, error_log_path, llm_data):
    """Process queries.jsonl and generate queries_llm.jsonl while logging errors."""
    errors = []
    with open(jsonl_input_path, "r", encoding="utf-8") as infile, open(
        jsonl_output_path, "w", encoding="utf-8"
    ) as outfile, open(error_log_path, "w", encoding="utf-8") as errorfile:
        for line in infile:
            query = json.loads(line)
            query_id = query["id"]

            if query_id in llm_data:
                query["text"] = llm_data[query_id]
            else:
                query["text"] = ""  # Leave text empty if no corresponding ID found
                errors.append(query_id)  # Log error

            query["sentence_annotations"] = []
            query["title"] = ""

            json.dump(query, outfile)
            outfile.write("\n")  # Ensure newline after each json object

        if errors:
            for error_id in errors:
                errorfile.write(f"{error_id}\n")


if __name__ == "__main__":
    # the raw TSV file generated from genrate_gpt_queries.py
    tsv_path = "<SET PATH>"

    # the queries.jsonl file from the TREC dataset
    jsonl_input_path = "<SET PATH>"

    # generate our queries file following the TREC format
    jsonl_output_path = "<SET PATH>"

    # the error log file
    error_log_path = "<SET PATH>"

    llm_data = load_llm_data(tsv_path)
    process_queries(jsonl_input_path, jsonl_output_path, error_log_path, llm_data)

    print("Processing complete. Check queries_llm.jsonl and error.log for output.")
