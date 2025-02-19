import csv
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import logging
from collections import Counter
import re
from transformers import GPT2Tokenizer

# Load OpenAI key from the .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Static variable TEMPLATE for prompting
TEMPLATE_0 = """
Let’s say I forgot the name of a {topic}, {ToTObject}, and I'm trying to find the name by writing a verbose post to an online community forum. In the post, I will describe what I remember about {ToTObject}. The post can include:
1. My own contextual memories from the past, e.g., "When I was young, …"
2. Comparisons with other people,
3. Mentions of uncertainty, e.g., "I’m not sure if it is true, but …"
4. Factually false memories,
5. Mentions of exclusion criteria, e.g., "its name is definitely not ..."

Pretend that you completely forgot the {topic} name ({ToTObject}), and write a verbose post that other community members can help you recall the name. Here are some rules when writing the post:
1. Never mention the exact name {ToTObject}.
2. Do not write a post like a letter - No "hello" and "goodbye".
3. Everyone who reads the post assumes that you forgot the name. Do not write long lines describing you forgot that. Go straight to explaining it.
4. Write about 200 words.
"""

TEMPLATE_1 = """
Let’s say I forgot the name of a {topic}, {ToTObject}, and I'm trying to find the name by writing a verbose post to an online community forum. In the post, I will describe what I remember about {ToTObject}. The post can include:
1. My own contextual memories from the past, e.g., "When I was young, …"
2. Comparisons with other people,
3. Mentions of uncertainty, e.g., "I’m not sure if it is true, but …"
4. Factually false memories,
5. Mentions of exclusion criteria, e.g., "its name is definitely not ..."

Here is some information about the {topic}, {ToTObject}:
Information: {Psg}

Pretend that you completely forgot the {topic} name ({ToTObject}), and write a verbose post that other community members can help you recall the name. Here are some rules when writing the post:
1. Never mention the exact name {ToTObject}.
2. Do not write a post like a letter - No "hello" and "goodbye".
3. Everyone who reads the post assumes that you forgot the name. Do not write long lines describing you forgot that. Go straight to explaining it.
4. Write about 200 words.
"""


TEMPLATE_2 = """
You are a user on an online community forum seeking help to recall a specific {topic} name on the tip of your tongue.
You are now writing a post to describe the {topic} {ToTObject} in detail to get help from other community members.
I am providing you with the following information about {ToTObject}, and you can use it to include some details in your post:
{Psg}

**important**: {ToTObject} is only provided to you to write the post. You should pretend that you forgot the name {ToTObject} and never mention it in the post!

Here are some example posts from other users, you can follow their style:

Example 1:
"I remember watching this movie back in the 90s in english this boy and a man was running from something. it seemed like a horror movie. only scene i can recall in my head is the man and boy was in a house or shed and these sharp eyeball looking things but were metal and had blades on the side that were trying to kill them. the blades went after the boy i think and he ducked and it went onto the door and got stuck."

Example 2:
"A movie I saw on tv a while ago about a few girls, maybe siblings or siblings and friends, who perform a witchy ritual in the woods (i think they were naked) and 1 or maybe 2 girls fell into a coma. i remember that she was in her bed in an attic room as well. I dont remember if it was american or british but it was in english. it was a fairly recent movie, i think maybe 2000s but not before 1995 at least. please help i really want to watch this movie!"

Example 3:
"At some point the the 90's I saw a film on tv (probably HBO) that was set in the 1700's. I don't remember much, but I was just thinking about it and it's bothering me not knowing now. I remember there was a decent amount of sex/sexual references. It's not your typical pride and prejudiced, sense and sensibility type period piece. I'm pretty sure there were quite a few well known actors in it as well, I just can't seem to remember any of them. There was a point in the film where there was some sort of scheme for this one man to sleep with a young girl and take her virginity. That's all I really remember. Any ideas?"

Example 4:
"I am looking for a 1940s-1950s black and white movie that I have seen on TV a long time ago (15-20 years ago). I think it was a dark comedy, with an opening scene featuring two women hitchhiking; one of them lifts her skirts and makes a car crash. They reach a mansion (which was about to be inherited by one of them), and spend the night. One of the women gets killed (the one inheriting the mansion was supposed to get killed, but because they changed rooms, the other one gets murdered). The ghost of the dead woman remains to haunt the house. A clumsy detective appears to solve the crime but gets himself in a set of ridiculous situations (at some point there is a scene in which he gets trapped in a refrigerator); in the attempt to solve the mystery, they discover secret passageways and chambers of the mansion (one of them leads to a river used to transport dead bodies). That's all the information I have (and I am not entirely certain of their accuracy - it's been a long time since I've seen the movie). Any suggestions are much appreciated. Cheers!"

Example 5:
"I saw this presidential spoof movie on TCM about 6 years ago, most likely originally from the 70s. It was in color. I remember two scenes. In one scene the protagonist is on a house boat I think it was yellow. I believe he may have been choosing clothes at some point. I remember him talking to someone as he walked throughout the boat. In another scene I think he tries to reach out to hippies. He tries drugs and makes love in the grass with one maybe two women. I saw this on TV so nothing explicit, but I remember the humor being very dry and adult."

Example 6:
"I can't remember what the overall theme was but there was a scene in it where a man and woman were in the backseat of a car making out etc. The view switched back and forth from what they were doing to a cartoon of what was happening inside the man's body, like, little guys pulling levers. Anyone remember what the movie was?"
"""


TEMPLATE_3 = """
You are a user on an online community forum seeking help to recall a specific {topic} name on the tip of your tongue.
You are now writing a post to describe the {topic} {ToTObject} in detail to get help from other community members.

In the post, you are going to describe what you remember about {ToTObject}. The post can include:
1. Your own contextual memories from the past, e.g., "When I was young, …"
2. Comparisons with other people,
3. Mentions of uncertainty, e.g., "I'm not sure if it is true, but …"
4. Factually false memories,
5. Mentions of exclusion criteria, e.g., "its name is definitely not …"

Pretend that you completely forgot the {topic} name ({ToTObject}), and write a verbose post that other community members can help you recall the name. Here are some rules when writing the post:
1. Never mention the exact name {ToTObject}.
2. Do not write a post like a letter - No "hello" and "goodbye".
3. Everyone who reads the post assumes that you forgot the name. Do not write long lines describing you forgot that. Go straight to explaining it.
4. Write about 200 words.

Here are some example posts from other users, you can follow their style:

Example 1:
"I remember watching this movie back in the 90s in english this boy and a man was running from something. it seemed like a horror movie. only scene i can recall in my head is the man and boy was in a house or shed and these sharp eyeball looking things but were metal and had blades on the side that were trying to kill them. the blades went after the boy i think and he ducked and it went onto the door and got stuck."

Example 2:
"A movie I saw on tv a while ago about a few girls, maybe siblings or siblings and friends, who perform a witchy ritual in the woods (i think they were naked) and 1 or maybe 2 girls fell into a coma. i remember that she was in her bed in an attic room as well. I dont remember if it was american or british but it was in english. it was a fairly recent movie, i think maybe 2000s but not before 1995 at least. please help i really want to watch this movie!"

Example 3:
"At some point the the 90's I saw a film on tv (probably HBO) that was set in the 1700's. I don't remember much, but I was just thinking about it and it's bothering me not knowing now. I remember there was a decent amount of sex/sexual references. It's not your typical pride and prejudiced, sense and sensibility type period piece. I'm pretty sure there were quite a few well known actors in it as well, I just can't seem to remember any of them. There was a point in the film where there was some sort of scheme for this one man to sleep with a young girl and take her virginity. That's all I really remember. Any ideas?"

Example 4:
"I am looking for a 1940s-1950s black and white movie that I have seen on TV a long time ago (15-20 years ago). I think it was a dark comedy, with an opening scene featuring two women hitchhiking; one of them lifts her skirts and makes a car crash. They reach a mansion (which was about to be inherited by one of them), and spend the night. One of the women gets killed (the one inheriting the mansion was supposed to get killed, but because they changed rooms, the other one gets murdered). The ghost of the dead woman remains to haunt the house. A clumsy detective appears to solve the crime but gets himself in a set of ridiculous situations (at some point there is a scene in which he gets trapped in a refrigerator); in the attempt to solve the mystery, they discover secret passageways and chambers of the mansion (one of them leads to a river used to transport dead bodies). That's all the information I have (and I am not entirely certain of their accuracy - it's been a long time since I've seen the movie). Any suggestions are much appreciated. Cheers!"

Example 5:
"I saw this presidential spoof movie on TCM about 6 years ago, most likely originally from the 70s. It was in color. I remember two scenes. In one scene the protagonist is on a house boat I think it was yellow. I believe he may have been choosing clothes at some point. I remember him talking to someone as he walked throughout the boat. In another scene I think he tries to reach out to hippies. He tries drugs and makes love in the grass with one maybe two women. I saw this on TV so nothing explicit, but I remember the humor being very dry and adult."

Example 6:
"I can't remember what the overall theme was but there was a scene in it where a man and woman were in the backseat of a car making out etc. The view switched back and forth from what they were doing to a cartoon of what was happening inside the man's body, like, little guys pulling levers. Anyone remember what the movie was?"
"""


TEMPLATE_4 = """
Let's do a role play. You are now a person who watched a movie {ToTObject} a long time ago and forgot the movie's name. You are trying to recall the name by posting a verbose post in an online forum like Reddit describing the movie. Generate a post of length of about 200 words about the movie {ToTObject}. Your post must describe a vague memory of a movie without mentioning its exact name. People in the forum must have a hard time figuring out which movie you are looking for. The answer should be hard to find in search engines, so do not write too obvious search terms. I will provide you a basic information about the movie, and you have to follow the guidelines to generate a post.

Information about {ToTObject}:
{Psg}

Guidelines:
1. Include personal context, such as "When I was young" or "I remember watching this with my family".
2. Describe sensory details like the atmosphere, sounds, or feelings the movie evoked.
3. Compare the movie to other movies or familiar scenarios.
4. Express uncertainty to reflect the imperfect nature of memory, e.g., "I'm not sure if it is true, but".
5. Mention factually false memories to add authenticity and to make it a little harder.
6. Avoid mentioning the exact movie or actor names.
7. Describe specific scenes or moments from the movie.
8. When mentioning the characters, avoid use their exact name. Blur the name use or generic names like "the main character", "the man/woman", "the parents/kids", "the father/daughter", and "the villain", etc.
9. Indicate when you watched it and where, e.g., "I saw this movie about 10 years ago on TV at home".
10. Write in a casual, conversational tone without formal greetings or farewells.
11. Provide enough detail to make the memory vivid but leave room for interpretation.
12. Invite engagement from other community members, e.g., "Any ideas?" or "Does anyone remember this?"
13. Do not copy-paste the information or the above example provided, use them as a reference to generate a unique post.

Generate a post based on these guidelines.
"""


TEMPLATE_5 = """
Let's do a role play. You are now a person who watched a movie {ToTObject} a long time ago and forgot the movie's name. You are trying to recall the name by posting a verbose post in an online forum like Reddit describing the movie. Generate a post of length of about 200 words about the movie {ToTObject}. Your post must describe a vague memory of a movie without mentioning its exact name. People in the forum must have a hard time figuring out which movie you are looking for. The answer should be hard to find in search engines, so do not write too obvious search terms. I will provide you a basic information about the movie, and you have to follow the guidelines to generate a post.

Information about {ToTObject}:
{Psg}

Guidelines:
1. Share a personal anecdote related to when or with whom you watched the movie, but avoid common phrases like "When I was young". Instead, think of unique ways to set the scene.
2. Focus on sensory details such as the overall mood, sounds, or emotional impact of the movie, using vivid descriptions.
3. Draw comparisons with other movies or familiar experiences but in a nuanced manner that doesn't directly echo well-known titles.
4. Reflect the imperfect nature of memory with phrases that express doubt or mixed recollections, avoiding direct phrases like "I'm not sure if it is true, but".
5. Introduce a few incorrect or mixed-up details to make the recollection seem more realistic and challenging to pinpoint.
6. Do not specify any movie or actor names directly.
7. Describe particular scenes or moments using ambiguous terms or partial descriptions.
8. Refer to characters in a non-specific way using descriptions or roles rather than names.
9. Mention vaguely when and where you watched the movie, and encourage using less typical references than "10 years ago on TV".
10. Maintain a casual and conversational tone throughout the post, ensuring it sounds natural and engaging without using formal structures.
11. Provide vivid but ambiguous details to stir the reader's imagination while leaving them guessing.
12. Encourage responses with questions or prompts for help that sound genuine and open-ended.
13. Use the provided examples only as inspiration to craft a unique and engaging narrative, avoiding any direct replication of sample phrases.
14. Avoid using formal greetings such as "Hello" or "Hey everyone," and start directly with your post.

Generate a post based on these enriched guidelines.
"""


TEMPLATE_6 = """
Let's do a role play. You are now a person who watched a movie {ToTObject} a long time ago and forgot the movie's name. You are trying to recall the name by posting a verbose post in an online forum like Reddit describing the movie. Generate a post of length of about 200 words about the movie {ToTObject}. Your post must describe a vague memory of a movie without mentioning its exact name. People in the forum must have a hard time figuring out which movie you are looking for. The answer should be hard to find in search engines, so do not write too obvious search terms. I will provide you a basic information about the movie, and you have to follow the guidelines to generate a post.

Information about {ToTObject}:
{Psg}

Guidelines:
MUST FOLLOW:
1. Reflect the imperfect nature of memory with phrases that express doubt or mixed recollections, avoiding direct phrases like "I'm not sure if it is true, but".
2. Do not specify any movie or actor names directly.
3. Refer to characters in a non-specific way using descriptions or roles rather than names.
4. Maintain a casual and conversational tone throughout the post, ensuring it sounds natural and engaging without using formal structures.
5. Provide vivid but ambiguous details to stir the reader's imagination while leaving them guessing.
6. Use the provided examples only as inspiration to craft a unique and engaging narrative, avoiding any direct replication of sample phrases.
7. Avoid using formal greetings such as "Hello" or "Hey everyone," and start directly with your post.

COULD FOLLOW:
1. Share a personal anecdote related to when or with whom you watched the movie, but avoid common phrases like "When I was young". Instead, think of unique ways to set the scene.
2. Focus on sensory details such as the overall mood, sounds, or emotional impact of the movie, using vivid descriptions.
3. Draw comparisons with other movies or familiar experiences but in a nuanced manner that doesn't directly echo well-known titles.
4. Introduce a few incorrect or mixed-up details to make the recollection seem more realistic and challenging to pinpoint.
5. Describe particular scenes or moments using ambiguous terms or partial descriptions.
6. Mention vaguely when and where you watched the movie, and encourage using less typical references than "10 years ago on TV".
7. Encourage responses with questions or prompts for help that sound genuine and open-ended.

Generate a post based on these guidelines.
"""


TEMPLATE_CELEB = """
Let's do a role play. You are now a person who vaguely remembers a public figure called {ToTObject}, but forgot the person's name. You are trying to recall the name by posting a verbose post in an online forum like Reddit describing the person. Generate a post of around 200 words about the person {ToTObject}. Your post must describe a vague memory of the person without revealing its exact name. People on the forum must have a hard time figuring out which person you are looking for. The answer should be difficult to find in search engines, so avoid using obvious keywords. I will provide you with some basic information about the person, and you must follow the guidelines to create a post.

Information about {ToTObject}:
{Psg}

Guidelines:
MUST FOLLOW:
1. Reflect the imperfect nature of memory with phrases that express doubt or mixed recollections, avoiding direct phrases like "I'm not sure if it is true, but".
2. Do not directly specify the name of the person.
3. Refer to the person in an ambiguous way using descriptions instead of names.
4. Maintain a casual and conversational tone throughout the post, making sure it sounds natural and engaging without using formal structures.
5. Provide vivid but ambiguous details to stir the reader's imagination while keeping them guessing.
6. Use the provided information only as inspiration to craft a unique and engaging narrative, avoiding any direct replication of the given phrases.
7. Start directly with your post, avoiding formal greetings like "Hello" or "Hey everyone."
8. Start directly with your post, without describing your state of mind like "So, there's this", "I remember", "I've been thinking about".

COULD FOLLOW:
1. Share a personal anecdote related to the person, but avoid common phrases like "When I was young." Instead, find unique ways to set the scene.
2. Draw comparisons with other similar public figures in a nuanced way that doesn't directly echo well-known people.
3. Introduce a few incorrect or mixed-up details to make the recollection seem more realistic and harder to pinpoint.
4. Describe particular scenes or moments using ambiguous terms or partial descriptions.
5. End the post by encouraging responses with genuine, open-ended questions for help.

Generate a post based on these guidelines.
"""


TEMPLATE_LANDMARK = """
Let's do a role play. You are now a person who vaguely remembers a place called {ToTObject}. You are trying to recall the name of the place by posting a verbose post in an online forum like Reddit describing the place. Generate a post of around 200 words about the place {ToTObject}. Your post must describe a vague memory of the place without revealing its exact name. People on the forum must have a hard time figuring out which place you are looking for. The answer should be difficult to find in search engines, so avoid using obvious keywords. I will provide you with some basic information about the place, and you must follow the guidelines to create a post.

Information about {ToTObject}:
{Psg}

Guidelines:
MUST FOLLOW:
1. Reflect the imperfect nature of memory with phrases that express doubt or mixed recollections, avoiding direct phrases like "I'm not sure if it is true, but".
2. Do not directly specify the name of the place.
3. Refer to the places in an ambiguous way using descriptions instead of names.
4. Maintain a casual and conversational tone throughout the post, making sure it sounds natural and engaging without using formal structures.
5. Provide vivid but ambiguous details to stir the reader's imagination while keeping them guessing.
6. Use the provided information only as inspiration to craft a unique and engaging narrative, avoiding any direct replication of the given phrases.
7. Start directly with your post, avoiding formal greetings like "Hello" or "Hey everyone."
8. Start directly with your post, without describing your state of mind like "So, there's this", "I remember", "I've been thinking about".

COULD FOLLOW:
1. Share a personal anecdote about your time at the place and the people you were with, but avoid common phrases like "When I was young." Instead, find unique ways to set the scene.
2. Focus on sensory details like the overall mood, sounds, and emotional impact of being in the place, using vivid descriptions.
3. Draw comparisons with other places or familiar experiences in a nuanced way that doesn't directly echo well-known locations.
4. Introduce a few incorrect or mixed-up details to make the recollection seem more realistic and harder to pinpoint.
5. Describe particular scenes or moments using ambiguous terms or partial descriptions.
6. End the post by encouraging responses with genuine, open-ended questions for help.

Generate a post based on these guidelines.
"""


def fetch_document_from_Wiki_intro(tgt_object):
    # Construct the URL for the Wikipedia API to fetch the full content
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&titles={tgt_object}&exintro"

    # Make the request to Wikipedia API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Ensure the 'pages' dictionary exists and has content
        if data.get("query", {}).get("pages"):
            # Extract the page content from the response
            page = next(
                iter(data["query"]["pages"].values()), None
            )  # Get the first page object, if any
            if page and "extract" in page:
                # Use BeautifulSoup to clean the HTML content
                soup = BeautifulSoup(page["extract"], "html.parser")
                # Extract and clean paragraph texts
                paragraphs = [
                    para.get_text(" ", strip=True) for para in soup.find_all("p")
                ]
                # Filter out empty paragraphs and return the result
                return [para for para in paragraphs if para]
            else:
                return ["Failed to retrieve content from the page."]
        else:
            return ["No pages found in the API response."]
    else:
        # Return a message if there was a problem with the request
        return ["Failed to retrieve content from Wikipedia."]


def fetch_document_from_Wiki_intro_and_plot(tgt_object):
    # Construct the URL for the Wikipedia API to fetch the full content without only intro
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&titles={tgt_object}&explaintext=true"

    # Make the request to Wikipedia API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Ensure the 'pages' dictionary exists and has content
        if data.get("query", {}).get("pages"):
            # Extract the page content from the response
            page = next(iter(data["query"]["pages"].values()), None)
            if page and "extract" in page:
                text = page["extract"]
                # Check for the presence of a Plot section
                plot_start = text.find("== Plot ==")
                if plot_start != -1:
                    # Extract everything up to Plot
                    intro_text = text[:plot_start].strip()
                    # Extract the Plot content
                    plot_end = text.find("==", plot_start + 10)
                    if plot_end == -1:
                        plot_text = text[plot_start + 10 :].strip()
                    else:
                        plot_text = text[plot_start + 10 : plot_end].strip()
                    full_text = intro_text + plot_text
                else:
                    # Find the first section start if no Plot section
                    first_section_start = text.find("==")
                    if first_section_start != -1:
                        # Return content only up to the first section
                        full_text = text[:first_section_start].strip()
                    else:
                        # Return all text if no sections found
                        full_text = text.strip()
                return full_text
            else:
                return "Failed to retrieve content from the page."
        else:
            return "No pages found in the API response."
    else:
        # Return a message if there was a problem with the request
        return "Failed to retrieve content from Wikipedia."


def fetch_document_from_Wiki_intro_full(tgt_object):
    # Construct the URL for the Wikipedia API to fetch the full content without only intro
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&titles={tgt_object}&explaintext=true"

    # Make the request to Wikipedia API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Ensure the 'pages' dictionary exists and has content
        if data.get("query", {}).get("pages"):
            # Extract the page content from the response
            page = next(iter(data["query"]["pages"].values()), None)
            if page and "extract" in page:
                text = page["extract"]
                full_text = text.strip()

                return full_text
            else:
                return "Failed to retrieve content from the page."
        else:
            return "No pages found in the API response."
    else:
        # Return a message if there was a problem with the request
        return "Failed to retrieve content from Wikipedia."


def split_document(document, max_paragraphs=5):
    """
    Splits the document (a list of paragraphs) to return only the first few paragraphs.
    """
    # Split the document into paragraphs based on newlines
    paragraphs = document.split("\n")
    # Filter out any empty strings that may occur due to consecutive newline characters
    # return [para.strip() for para in paragraphs if para.strip()][:max_paragraphs]
    return [para.strip() for para in paragraphs if para.strip()]


def generate_post_without_name(template, topic, tgt_object, paragraphs):
    """
    Generates a post about the topic without mentioning its name, based on the template.
    """
    formatted_paragraphs = "\n".join(
        ["- " + para for para in paragraphs]
    )  # Format paragraphs for display
    prompt = template.format(
        ToTObject=tgt_object, topic=topic, Psg=formatted_paragraphs
    )

    messages = [
        # {"role": "system", "content": "You are a user on an online forum and want to ask a movie name on the tip of your tongue."},
        # {"role": "system", "content": "You are a user on an online forum and want to ask a landmark name on the tip of your tongue."},
        {
            "role": "system",
            "content": "You are a user on an online forum and want to ask a celebrity name on the tip of your tongue.",
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=1024, temperature=0.3
    )

    return response.choices[0].message.content.strip()


def summarize_paragraphs(paragraphs):
    """
    Summarizes the given input paragraphs into a two-paragraph summary.
    """
    input_text = "\n\n".join(paragraphs)
    # prompt = f"Please summarize the following description about a movie into two paragraphs:\n\n{input_text}. Please focus on the plots, and ignore the director and actor names."

    prompt = f"Please summarize the following description about a person into two paragraphs:\n\n{input_text}."

    # prompt = f"Please summarize the following description about a place into two paragraphs:\n\n{input_text}."

    messages = [
        {"role": "system", "content": "You are a text summarization assistant."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=1024, temperature=0.5
    )

    return response.choices[0].message.content.strip()


def summarize_paragraphs_truncate_after_max_tokens(paragraphs, max_tokens=128000):
    """
    Summarizes the given input paragraphs into a two-paragraph summary.
    Ensures that the input length in terms of tokens does not exceed the specified maximum.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Load a GPT-2 tokenizer

    # Combine paragraphs into a single string
    input_text = "\n\n".join(paragraphs)

    # Tokenize the input and check the token count
    tokens = tokenizer.encode(
        input_text, truncation=True, max_length=max_tokens, return_tensors="pt"
    )

    # Convert tokens back to text ensuring it does not exceed the max tokens
    truncated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)

    prompt = f"Please summarize the following description about a person into two paragraphs:\n\n{truncated_text}."

    messages = [
        {"role": "system", "content": "You are a text summarization assistant."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=1024, temperature=0.5
    )

    return response.choices[0].message.content.strip()


def setup_logging(file_path):
    log_folder = file_path  # Specify the folder where you want to save the log files
    os.makedirs(log_folder, exist_ok=True)  # Ensure the log folder exists

    # Setup basic configuration for logging, adjust the filename to include the folder path
    logging.basicConfig(
        filename=os.path.join(log_folder, "process_log.txt"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Logger for tracking targets that didn't pass the check after 3 retries
    failed_logger = logging.getLogger("failed_objects")
    fh = logging.FileHandler(
        os.path.join(log_folder, "failed_objects_log.txt")
    )  # Adjust the filename to include the folder path
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    failed_logger.addHandler(fh)
    return failed_logger


def load_query_ids(file_path):
    ids = set()
    with open(file_path, "r", encoding="utf-8") as json_file:
        for line in json_file:
            data = json.loads(line)
            ids.add(data["id"])
    return ids


def load_ids_from_tsv(file_path):
    ids = set()
    with open(file_path, "r", encoding="utf-8") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)  # Skip header
        for row in reader:
            ids.add(row[0])
    return ids


def check_missing_ids(jsonl_path_reference, tsv_path_generated):
    jsonl_ids = load_query_ids(jsonl_path_reference)
    tsv_ids = load_ids_from_tsv(tsv_path_generated)

    missing_in_tsv = jsonl_ids - tsv_ids  # IDs in JSONL but not in TSV
    missing_in_jsonl = tsv_ids - jsonl_ids  # IDs in TSV but not in JSONL

    if missing_in_tsv:
        print(f"IDs in reference JSONL but missing in generated TSV: {missing_in_tsv}")
    else:
        print("No IDs are missing in TSV that are present in JSONL.")

    if missing_in_jsonl:
        print(
            f"IDs in generated TSV but missing in reference JSONL: {missing_in_jsonl}"
        )
    else:
        print("No IDs are missing in JSONL that are present in TSV.")

    return missing_in_tsv, missing_in_jsonl


def check_duplicate_ids_in_tsv(file_path):
    ids = []
    with open(file_path, "r", encoding="utf-8") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)  # Skip the header
        for row in reader:
            ids.append(row[0])

    id_counts = Counter(ids)
    duplicate_ids = {id: count for id, count in id_counts.items() if count > 1}

    if duplicate_ids:
        print("Duplicate IDs found:")
        for id, count in duplicate_ids.items():
            print(f"ID {id} appears {count} times.")
    else:
        print("No duplicate IDs found.")

    return duplicate_ids


def generate_multiple(
    input_file_path, output_file_path, json_folder, log_folder_path, queries_file_path
):

    # Ensure the directory for JSON files exists
    os.makedirs(json_folder, exist_ok=True)

    # Setup logging
    failed_logger = setup_logging(log_folder_path)

    query_ids = load_query_ids(queries_file_path)

    # Open the input file and read it
    with open(input_file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile, delimiter="\t")
        fieldnames = ["ID", "QuestionBody", "wikipediaURL", "movieName"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        # Process the lines for testing
        for i, row in enumerate(reader):
            tgt_id = row["ID"]
            if tgt_id not in query_ids:
                continue

            retries = 0
            tgt_object = row["movieName"]  # Use the "movieName" as "tgt_object"
            tgt_id = row["ID"]

            # topic = 'movie'
            topic = "celebrity"

            try:
                # paragraphs = []  # without paragraphs

                # doc = fetch_document_from_Wiki_intro_and_plot(tgt_object)
                doc = fetch_document_from_Wiki_intro_full(tgt_object)
                print("doc-----------")
                print(doc)

                try:
                    summarization = summarize_paragraphs(doc)
                except Exception as e:
                    # Handle token limit error and retry with truncation
                    logging.info(
                        f"Token limit exceeded for {tgt_object}. Attempting to truncate..."
                    )
                    summarization = summarize_paragraphs_truncate_after_max_tokens(doc)

                print("summarization-----------")
                print(summarization)

                paragraphs = split_document(summarization)
                print("paragraphs-----------")
                print(paragraphs)

                if not paragraphs or not any(
                    paragraphs
                ):  # Check if paragraphs are empty
                    failed_logger.error(
                        f"ID: {tgt_id}, movieName: {tgt_object} has empty paragraphs."
                    )
                    continue  # Skip to the next row

                while retries < 3:
                    try:
                        response = generate_post_without_name(
                            TEMPLATE_CELEB, topic, tgt_object, paragraphs
                        )
                    except Exception as e:
                        failed_logger.error(
                            f"GPT API error for ID: {tgt_id}, movieName: {tgt_object}. Error: {e}"
                        )
                        break

                    if not response:  # Check if response is empty
                        failed_logger.error(
                            f"Empty response for ID: {tgt_id}, movieName: {tgt_object}."
                        )
                        break

                    if tgt_object.lower() in (response or "").lower():
                        retries += 1
                        logging.info(
                            f"Retrying for ID: {tgt_id}, movieName: {tgt_object} as the response contained the name. Attempt {retries}"
                        )
                        if retries >= 3:
                            logging.info(
                                f"Skipping ID: {tgt_id}, movieName: {tgt_object} after 3 retries."
                            )
                            failed_logger.error(
                                f"ID: {tgt_id}, movieName: {tgt_object} failed to generate valid response after 3 retries."
                            )
                            break  # Skip this object after max retries
                        continue

                    # Valid response found
                    processed_response = response.replace("\n", " ").strip('"')
                    row["QuestionBody"] = processed_response
                    # writer.writerow(row)
                    writer.writerow(
                        {k: row[k] for k in fieldnames}
                    )  # Write only the specified fields

                    result = {
                        "topic": topic,
                        "ToTObject": tgt_object,
                        "paragraphs": paragraphs,
                        "prompt": TEMPLATE_CELEB,
                        "response": response,
                    }
                    file_name = os.path.join(
                        json_folder, f'{tgt_object.replace(" ", "_")}.json'
                    )
                    with open(file_name, "w") as f:
                        json.dump(result, f, indent=4)

                    logging.info(
                        f"Processed {i+1} row with movie name {tgt_object}. Response: {response}"
                    )
                    break  # Exit loop after successful processing

            except Exception as e:
                failed_logger.error(
                    f"Failed to process ID: {tgt_id}, movieName: {tgt_object} due to an error: {e}"
                )

    logging.info("Updated TSV file with generated responses has been created.")


def generate_multiple_resume_from_index(
    input_file_path, output_file_path, json_folder, log_folder_path, queries_file_path
):
    start_row_index = 1466  # Adjust this to the row index from which to resume

    # Ensure the directory for JSON files exists
    os.makedirs(json_folder, exist_ok=True)

    # Setup logging
    failed_logger = setup_logging(log_folder_path)

    query_ids = load_query_ids(queries_file_path)

    # Open the input file and read it
    with open(input_file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "a", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile, delimiter="\t")
        fieldnames = ["ID", "QuestionBody", "wikipediaURL", "movieName"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")

        # Write the header only if the file is empty
        if os.stat(output_file_path).st_size == 0:
            writer.writeheader()

        # Process the lines starting from the specified row
        for i, row in enumerate(reader):
            if i < start_row_index:
                continue  # Skip rows until the start index is reached

            tgt_id = row["ID"]
            if tgt_id not in query_ids:
                continue

            retries = 0
            tgt_object = row["movieName"]  # Use the "movieName" as "tgt_object"
            tgt_id = row["ID"]

            # topic = 'movie'
            # topic = 'landmark'
            topic = "celebrity"

            try:

                # paragraphs = []  # without paragraphs

                # doc = fetch_document_from_Wiki_intro_and_plot(tgt_object)
                doc = fetch_document_from_Wiki_intro_full(tgt_object)
                print("doc-----------")
                print(doc)

                try:
                    summarization = summarize_paragraphs(doc)
                except Exception as e:
                    # Handle token limit error and retry with truncation
                    logging.info(
                        f"Token limit exceeded for {tgt_object}. Attempting to truncate..."
                    )
                    summarization = summarize_paragraphs_truncate_after_max_tokens(doc)

                print("summarization-----------")
                print(summarization)

                paragraphs = split_document(summarization)
                print("paragraphs-----------")
                print(paragraphs)

                if not paragraphs or not any(
                    paragraphs
                ):  # Check if paragraphs are empty
                    failed_logger.error(
                        f"ID: {tgt_id}, movieName: {tgt_object} has empty paragraphs."
                    )
                    continue  # Skip to the next row

                while retries < 3:
                    try:
                        response = generate_post_without_name(
                            TEMPLATE_CELEB, topic, tgt_object, paragraphs
                        )
                    except Exception as e:
                        failed_logger.error(
                            f"GPT API error for ID: {tgt_id}, movieName: {tgt_object}. Error: {e}"
                        )
                        break

                    if not response:  # Check if response is empty
                        failed_logger.error(
                            f"Empty response for ID: {tgt_id}, movieName: {tgt_object}."
                        )
                        break

                    if tgt_object.lower() in (response or "").lower():
                        retries += 1
                        logging.info(
                            f"Retrying for ID: {tgt_id}, movieName: {tgt_object} as the response contained the name. Attempt {retries}"
                        )
                        if retries >= 3:
                            logging.info(
                                f"Skipping ID: {tgt_id}, movieName: {tgt_object} after 3 retries."
                            )
                            failed_logger.error(
                                f"ID: {tgt_id}, movieName: {tgt_object} failed to generate valid response after 3 retries."
                            )
                            break  # Skip this object after max retries
                        continue

                    # Valid response found
                    processed_response = response.replace("\n", " ").strip('"')
                    row["QuestionBody"] = processed_response
                    # writer.writerow(row)
                    writer.writerow(
                        {k: row[k] for k in fieldnames}
                    )  # Write only the specified fields

                    result = {
                        "topic": topic,
                        "ToTObject": tgt_object,
                        "paragraphs": paragraphs,
                        "prompt": TEMPLATE_CELEB,
                        "response": response,
                    }
                    file_name = os.path.join(
                        json_folder, f'{tgt_object.replace(" ", "_")}.json'
                    )
                    with open(file_name, "w") as f:
                        json.dump(result, f, indent=4)

                    logging.info(
                        f"Processed {i+1} row with movie name {tgt_object}. Response: {response}"
                    )
                    break  # Exit loop after successful processing

            except Exception as e:
                failed_logger.error(
                    f"Failed to process ID: {tgt_id}, movieName: {tgt_object} due to an error: {e}"
                )

    logging.info("Updated TSV file with generated responses has been created.")


def generate_single(output_file_path):

    topic = "movie"  # Examples: 'celebrity', 'movie', 'landmark'

    mstot_id = "511"
    tgt_object = "Dancing on the Moon"
    wikipediaURL = "https://en.wikipedia.org/wiki/Dancing_on_the_Moon"

    doc = fetch_document_from_Wiki_intro_full(tgt_object)

    summarization = summarize_paragraphs(doc)
    paragraphs = split_document(summarization)

    response = generate_post_without_name(TEMPLATE_CELEB, topic, tgt_object, paragraphs)
    print(response)
    assert tgt_object.lower() not in (response or "").lower()

    # Save the generated post to a JSON file named after the target object
    result = {
        "topic": topic,
        "ToTObject": tgt_object,
        "paragraphs": paragraphs,
        "prompt": TEMPLATE_CELEB,
        "response": response,
    }
    file_name = f'{tgt_object.replace(" ", "_")}.json'
    with open(file_name, "w") as f:
        json.dump(result, f, indent=4)

    # Print the generated post
    print(response)

    processed_response = response.replace("\n", " ").strip('"')

    # Write the result to the output file
    with open(output_file_path, "a", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["ID", "QuestionBody", "wikipediaURL", "movieName"],
            delimiter="\t",
        )
        writer.writerow(
            {
                "ID": mstot_id,
                "QuestionBody": processed_response,
                "wikipediaURL": wikipediaURL,
                "movieName": tgt_object,
            }
        )


def generate_single_for_failed(mstot_id, tgt_object, output_file_path):
    """
    Process a single movie entry using its ID and name.
    """
    wikipediaURL = "N/A"  # Since URL is not used, we set it to 'N/A'

    # Fetch the document, summarize it, and generate the post
    doc = fetch_document_from_Wiki_intro_full(tgt_object)
    summarization = summarize_paragraphs_truncate_after_max_tokens(doc)
    paragraphs = split_document(summarization)
    response = generate_post_without_name(
        TEMPLATE_CELEB, "celebrity", tgt_object, paragraphs
    )

    # Ensure the target object name does not appear in the response
    assert tgt_object.lower() not in (response or "").lower()

    # Save the generated post to a JSON file named after the target object
    result = {
        "topic": "celebrity",
        "ToTObject": tgt_object,
        "paragraphs": paragraphs,
        "prompt": TEMPLATE_CELEB,
        "response": response,
    }
    file_name = f'{tgt_object.replace(" ", "_")}.json'
    with open(file_name, "w") as f:
        json.dump(result, f, indent=4)

    processed_response = response.replace("\n", " ").strip('"')

    # Write the result to the output file
    with open(output_file_path, "a", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["ID", "QuestionBody", "wikipediaURL", "movieName"],
            delimiter="\t",
        )
        writer.writerow(
            {
                "ID": mstot_id,
                "QuestionBody": processed_response,
                "wikipediaURL": wikipediaURL,
                "movieName": tgt_object,
            }
        )


def generate_from_failed_log(failed_log_path, output_file_path):
    """
    Reads a failed log file and retries processing for each failed entry.
    """
    with open(failed_log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            # Extract ID and movieName using regular expressions
            match = re.search(r"ID: (.*?), movieName: (.*?) due to an error", line)
            if match:
                mstot_id = match.group(1).strip()
                tgt_object = match.group(2).strip()

                # Attempt to reprocess the failed entry
                print(f"Reprocessing ID: {mstot_id}, movieName: {tgt_object}")
                generate_single_for_failed(mstot_id, tgt_object, output_file_path)


if __name__ == "__main__":
    ## step 1
    # first run this to generate most of the data
    # The TSV file path
    input_file_path = "<SET PATH>"
    # The output TSV file path
    output_file_path = "<SET PATH>"

    json_folder = "<SET PATH>"
    log_folder_path = "<SET PATH>"

    # select queries IDs from [train, dev, test]
    queries_file_path = "<SET PATH>"

    generate_multiple(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        json_folder=json_folder,
        log_folder_path=log_folder_path,
        queries_file_path=queries_file_path,
    )
    # generate_multiple_resume_from_index(input_file_path=input_file_path, output_file_path=output_file_path, json_folder=json_folder, log_folder_path=log_folder_path, queries_file_path=queries_file_path)

    ## step 2
    # based on the error log, run this to generate the missing data individually
    # The output TSV file path
    # output_file_path = '<SET PATH>'
    # generate_single(output_file_path)

    # # based on the error log, run this to generate the missing data all at once
    # # The output TSV file path
    # failed_log_path = '<SET PATH>'
    # output_file_path = '<SET PATH>'
    # generate_from_failed_log(failed_log_path, output_file_path)

    ## step 3
    # check missing and duplicated IDs
    # jsonl_file_reference = '<SET PATH>'
    # tsv_file_generated = '<SET PATH>'
    # # '884', '667' for test split
    # missing_ids = check_missing_ids(jsonl_file_reference, tsv_file_generated)
    # duplicate_ids = check_duplicate_ids_in_tsv(tsv_file_generated)
