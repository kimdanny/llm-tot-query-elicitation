import json


def clean_movie_names(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)
            cleaned_movies = []

            # Iterate through each movie name and remove the order number
            for movie in data["gpt_queries"]:
                # Split on the first dot and space, then take the rest of the string
                cleaned_name = (
                    ". ".join(movie.split(". ")[1:]) if ". " in movie else movie
                )
                cleaned_movies.append(cleaned_name)

            data["gpt_queries"] = cleaned_movies

            json.dump(data, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    input_raw_ranking_file = "<SET PATH>"
    output_cleaned_ranking_file = "<SET PATH>"

    clean_movie_names(input_raw_ranking_file, output_cleaned_ranking_file)
