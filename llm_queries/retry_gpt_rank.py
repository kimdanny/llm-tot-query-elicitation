import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def process_retry_ids(input_file, output_file, retry_ids):
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    # Adjust the ID matching to ensure correct type (string vs integer)
    retry_ids = set(map(str, retry_ids))  # Ensure all IDs in the retry list are strings
    retry_entries = [
        json.loads(line) for line in lines if json.loads(line)["id"] in retry_ids
    ]

    if not retry_entries:
        print(f"No entries found for the provided retry IDs: {retry_ids}.")
    else:
        print(f"Found {len(retry_entries)} entries for retry. Processing...")

    for entry in retry_entries:
        retry_query(entry, output_file)


def retry_query(input_data, output_file):
    query_id = input_data["id"]
    query_text = input_data["text"]
    print(f"Processing ID {query_id}...")

    prompt = f"""
    You are an expert in movies. You are helping someone recollect a movie name that is on the tip of their tongue.
    You respond to each message with a list of 20 guesses for the name of the movie being described.
    **important**: you only mention the names of the movies, one per line, sorted by how likely they are the correct movie with the most likely correct movie first and the least likely movie last.
    message: {query_text}
    """

    attempts = 0
    while attempts < 3:
        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=1024,
                temperature=0.5,
            )

            movie_names = response.choices[0].text.strip().split("\n")
            movie_names = [name for name in movie_names if name]

            output_data = {"id": query_id, "gpt_queries": movie_names[:20]}

            with open(output_file, "a", encoding="utf-8") as outfile:
                json.dump(output_data, outfile)
                outfile.write("\n")
            print(f"Success: Data appended for ID {query_id}")
            break  # Exit the loop if successful

        except BadRequestError as e:
            attempts += 1
            print(f"Retry {attempts} for ID {query_id} failed: {str(e)}")
            time.sleep(1)  # Sleep to avoid hitting rate limits after an error

    if attempts == 3:
        print(f"Skipping ID {query_id} after 3 failed attempts.")


if __name__ == "__main__":
    input_source_file = "<SET PATH>"
    output_ranking = "<SET PATH>"
    retry_ids = []  # Add IDs that need reprocessing here. E.g., retry_ids = [1065]

    process_retry_ids(input_source_file, output_ranking, retry_ids)
