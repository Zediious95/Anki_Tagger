import pandas as pd
import numpy as np
import re, sys, csv, os
import openai
import tiktoken
from openai import APIError, RateLimitError, APIConnectionError
import time, requests

MAX_POOR_MATCH_RUN = 10
MAX_TOKENS_PER_OBJ = 30000 #Change token count if you more or less cards to be assessed per learning objective

def set_api_key():
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    except KeyError:
        print("Set your OpenAI API key as an environment variable named 'OPENAI_API_KEY' eg In terminal: export OPENAI_API_KEY=your-api-key")

def handle_api_error(func):
    def wrapper(*args, **kwargs):
        t=5
        while True:
            try:
                return func(*args, **kwargs)
            except (RateLimitError, APIError, APIConnectionError):
                print(f'API Error. Waiting {t}s before retrying.')
                time.sleep(t)  # wait for 10 seconds before retrying
                t+=5
    return wrapper

def convert_to_np_array(s):
    return np.fromstring(s.strip("[]"), sep=",")

def load_emb(path):

    # Specify the data types for columns 0, 1, and 2
    column_dtypes = {0: str, 1: str, 2: int}

    # Read CSV file and interpret column types
    df = pd.read_csv(
        path,
        dtype=column_dtypes,
        converters={3: convert_to_np_array})

    return df

def vs(x, y):
    return np.dot(np.array(x), np.array(y))

def construct_prompt(obj,card):

    prompt = f"Task: Rate the relevance of the Anki card to the learning question on a scale from 0 to 100. Be aware that even partial information related to core concepts should receive a high score if it contributes to understanding the learning question.\n\
     Learning questions can be complex, and Anki cards may contain fragmented information that partially addresses the full scope of the question.\n\
Instructions: Focus on how well the Anki card addresses any key concepts or foundational knowledge needed to answer the learning question, even if it doesn’t answer it fully.\n\
Partial answers that provide important context or core definitions should receive a higher score for their contribution to understanding.\n\
Break down complex learning questions into sub-questions and evaluate how the Anki card contributes to any part of the question, giving special weight to fundamental concepts related to the topic.\n\
Format: Start your statement with 'Score: # (just the final score number. No need to say 90 out of 100. Just say Score: 90)' followed with the arguement for the score. Limit your response to less than 25 words.\n\
Examples:\n\
Learning Question 1: Describe the morphology, features, functions, and lifespan of blood cells such as erythrocytes and leukocytes.\n\
Interpretation of sub-Question 1: What is the lifespan of an erythrocyte? What is the morphology of an erythrocyte? What are the features and functions of an erythrocyte? What is the lifespan of a leukocyte? What is the morphology of a leukocyte? What are the features and functions of a leukocyte?\n\
Anki Card 1: Erythrocytes (RBCs) carry O2 to tissues and CO2 to lungs.\n\
How to score Anki Card 1: Score 100 Since the card directly addresses the function of erythrocytes, a core part of the learning question, it should receive a high score (90–100) even though it doesn’t mention morphology or lifespan. Function is a critical part of understanding blood cells.\n\
Anki Card 2: Do erythrocytes contain organelles? No.\n\
How to score Anki Card 2: Score 90 This card identifies a unique morphological feature of erythrocytes (the lack of organelles), which is essential for distinguishing them from other cells. While it doesn’t cover function or lifespan, it still provides key information about morphology. It should receive a high score for contributing important knowledge toward understanding the morphology of erythrocytes.\n\
Anki Card 3: What RBC pathology is characterized by formation and loss of membrane blebs over time? Hereditary spherocytosis.\n\
How to score Anki Card 2: Score 40 This card touches up a unique pathologic morphology of erythrocytes, but the questions focused on non-pathological characteristics.\n\
Learning Question 2: Interpret complete blood count (CBC) results and identify abnormal values for leukocytes, erythrocytes, and platelets.\n\
Interpretation of sub-Question 2: What is a complete blood count (CBC) test? What is a normal CBC result for leukocytes? What is a normal CBC result for erythrocytes? What is a normal CBC result for platelets? What is an abnormal CBC result for leukocytes? What is an abnormal CBC result for erythrocytes? What is an abnormal CBC result for platelets?\n\
How to score: If the Anki card defines what a CBC test is, it should receive a high score (e.g., 85–100), even if it doesn’t provide specific details about normal or abnormal values for each cell type.\n\
Learning Question 3: What are the sequential steps involved in the processes of platelet activation and aggregation during primary hemostasis?\n\
Interpretation of sub-question 3: What are the steps involved in platelet activation? What are the steps involved in platelet aggregation? What is primary hemostasis?\n\
Anki Card 1: Following platelet adhesion (primary hemostasis), there is platelet degranulation.\n\
How to score Anki Card 1: Score 90 This card contains key terminology (platelet adhesion and degranulation) that is directly relevant to the sequence of events in primary hemostasis. It provides crucial information about the process and should be rated highly, even though it doesn’t describe all the steps in detail.\n\
Anki card 2: In order to adhere to the damaged endothelium, platelets bind vWF using the GPIb receptor\n\
How to score Anki Card 2: Score 85 This card provides critical information about the initial step in platelet adhesion (binding to von Willebrand factor via the GPIb receptor), which is part of the sequence in primary hemostasis. It should receive a high score as it describes a key step in the process of platelet activation and aggregation, even though it doesn’t cover all subsequent steps.\n\
Anki Card 3: Platelets (thrombocytes) are small cytoplasmic fragments derived from megakaryocytes.\n\
How to score Anki Card 3: Score 60 This card mentions platelets and it's origin, but it doesn't explore platelet activation, aggregation or primary homestasis.\n\
Anki Card 4: What is the most common cause of abnormal hemostasis in patients with chronic renal failure? Platelet dysfunction\n\
How to score Anki Card 4: Score 40 This card covers the abnormality of homestasis though platelet dysfunction, but it doesn't directly answer any question relating to platelet activation, aggregation or primary homestasis\n\
Learning Question 4: In what ways does melanization act as a protective strategy for the skin against harmful effects of ultraviolet (UV) light exposure?\n\
Interpretation of Question 4: What is melanization? How do melanocytes produce melanin? How does melanin protect the skin from UV light? What are the harmful effects of UV light?\n\
Anki Card 1: “Melanocytes synthesize melanin using the amino acid tyrosine as a precursor molecule.”\n\
How to score Anki Card 1: Score 90 Since the card provides essential context about melanin production by melanocytes, which is a core concept in answering how melanization protects against UV light, it should receive a high score for its relevance. Even though it doesn’t fully explain the protective strategy, it addresses foundational knowledge essential for understanding the broader mechanism.\n\
End of examples\n\
    Learning question: {obj}\n\
    Anki card: {card}"

    formatted_prompt = [{"role": "system", "content": "You are an assistant that precisely follows instructions."},
                        {"role": "user", "content": prompt}]

    return formatted_prompt

def count_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = list(enc.encode(text))
    return len(tokens)

def tokens_in_prompt(formatted_prompt):
    formatted_prompt_str = ""
    for message in formatted_prompt:
        formatted_prompt_str += message["content"] + " "
    #token_count = count_tokens(formatted_prompt_str)
    #print(f"Tokens in prompt: {token_count}")
    return count_tokens(formatted_prompt_str)

@handle_api_error
def rate_card_for_obj(prompt, temperature=1):
    # Calculate the remaining tokens for the response
    #remaining_tokens = 16000 - tokens_in_prompt(prompt) - 20
    remaining_tokens = 16000 - 20
    completions = openai.chat.completions.create(
        model="gpt-4o-mini",  # Use the gpt-4o-mini engine
        messages=prompt,
        max_tokens=remaining_tokens,  # Set the remaining tokens as the maximum for the response
        n=1,
        stop=None,
        temperature=temperature)

    string_return = completions.choices[0].message.content.strip()
    return string_return.replace('\n',' ')

def clean_reply(s):

    matches = re.search(r'Score: (\d{1,3})', s)

    if matches:
        score = matches.group(1)
        return int(score)

    else:
        matches = re.findall(r'\b([0-9][0-9]?|100)\b', s)
        if matches:
            numbers = [int(num) for num in matches]
            return min(numbers)
        else:
            return "NA"

def main(emb_path,obj_path):

    output_prefix = os.path.basename(obj_path).replace("_learning_objectives.csv",'')

    # load previous progress if exists
    last_processed_index = -1
    progress_file = f"{output_prefix}_progress.csv"

    if os.path.exists(progress_file):
        last_progress_df = pd.read_csv(progress_file)
        if not last_progress_df.empty:
            last_processed_index = last_progress_df.iloc[-1][0]

    emb_df = load_emb(emb_path)
    obj_df = load_emb(obj_path)

    with open(f'{output_prefix}_cards.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        if last_processed_index == -1:  # if there's no previous progress
            csv_writer.writerow(['guid','card','tag','cosine_sim','gpt_reply','score','objective'])

        for obj_index,obj_row in obj_df.iterrows():

            if obj_index <= last_processed_index:
                continue  # skip if the row has already been processed

            print(f"Processing objective {obj_index}")
            tag = obj_row['name']
            obj = obj_row['learning_objective']
            tokens = obj_row['tokens']
            obj_emb = obj_row['emb']

            emb_df["cosine_sim"] = emb_df.emb.apply(lambda x: vs(obj_emb,x))
            emb_df.sort_values(by='cosine_sim', ascending=False, inplace=True)

            poor_match_run_count = 0
            tokens_used = 0

            for index,emb_row in emb_df.iterrows():

                if poor_match_run_count > MAX_POOR_MATCH_RUN or tokens_used > MAX_TOKENS_PER_OBJ:
                    #print(f"Tokens used: {tokens_used}")
                    break

                guid = emb_row['guid']
                card = emb_row['card']
                cosine_sim = emb_row["cosine_sim"]
                gpt_reply = "NA"
                score = "NA"

                prompt = construct_prompt(obj,card)
                tokens_used += tokens_in_prompt(prompt)
                #print(f"Poor matches: {poor_match_run_count}")

                #try with progressively more creative juice
                temp = 0
                while score == "NA" and temp <= 1:
                    gpt_reply = rate_card_for_obj(prompt, temperature=temp)
                    score = clean_reply(gpt_reply)
                    temp += 0.25

                csv_writer.writerow([guid,card,tag,cosine_sim,gpt_reply,score,obj])
                if score > 50:
                    poor_match_run_count=0
                else:
                    poor_match_run_count+=1

            with open(progress_file, 'a', newline='', encoding='utf-8') as progress_csvfile:
                progress_csv_writer = csv.writer(progress_csvfile)
                progress_csv_writer.writerow([obj_index])

if __name__ == "__main__":
    set_api_key()
    if len(sys.argv) != 3:
        print("Usage: select_cards.py <deck_embeding> <learning_objectives>")
        sys.exit(1)
    emb_path = sys.argv[1]
    obj_path = sys.argv[2]
    main(emb_path,obj_path)
