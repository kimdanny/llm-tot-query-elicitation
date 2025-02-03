import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_movie_queries(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)
            query_id = data["id"]
            query_text = data["text"]

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
                        temperature=0.5,  # keep using 0.5 in GPT-based ranking for consistency
                    )

                    movie_names = response.choices[0].text.strip().split("\n")
                    movie_names = [name for name in movie_names if name]

                    output_data = {
                        "id": query_id,
                        "gpt_queries": movie_names[
                            :20
                        ],  # Limit to 20 movie names, follow TREC style
                    }

                    json.dump(output_data, outfile)
                    outfile.write("\n")
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

    generate_movie_queries(input_source_file, output_ranking)
