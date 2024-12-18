import os, re, sys, csv, glob, time
import openai
import tiktoken
import pdfplumber
from openai import RateLimitError, APIError
from util.embeddings_utils import get_embedding
from pathlib import Path

MAX_TOKENS = 16000
TOKEN_BUFFER = 2000


def set_api_key():
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    except KeyError:
        print(
            "Set your OpenAI API key as an environment variable named 'OPENAI_API_KEY' eg In terminal: export OPENAI_API_KEY=your-api-key")


def handle_api_error(func):
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except APIError as e:
                if isinstance(e, RateLimitError):
                    print(f"RateLimitError occurred: {e}")
                else:
                    print(f"APIError occurred: {e}")
                print(f"Error code: {e.code}")
                time.sleep(10)  # wait for 10 seconds before retrying

    return wrapper


def count_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = list(enc.encode(text))
    return len(tokens)


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text_pages = [page.extract_text() or "" for page in pdf.pages]
    return text_pages


def generate_questions(prompt, temperature=1.0):
    system_message = ("You are receiving lecture material for a medical school lesson. Use the following principles when making learning objectives (LO).\n\n"
                      "Material: \"Source Material\"\n\n"
                      "Task: Your task is to analyze the Source Material and condense the information into concise and direct learning objectives. Ensure that learning objectives are clearly written at a level appropriate for medical students while being easily understandable, and adheres to the specified formatting and reference criteria.\n\n"
                      "Formatting Criteria:\n"
                      "- Each LO needs to be succinct, precise, and exactly aligns with the Source Material.\n"
                      "- Each LO must be unique, with minimal overlapping information and stand on their own.\n"
                      "- Limit the word count of each Learning Objective to single sentences with less than 25 words.\n"
                      "- If a abbreviation or acronym is mentioned, include both the long name and the short name in each LO.\n\
                      Reference Criteria:\n"
                      "- Each LO must include at most two key terms on pathogenesis, pathophysiology, symptoms, treatments, diagnostics, etc., if mentioned in the Source Material.\n"
                      "- If material is briefly mentioned as a roadmap and/or stated to be explored in another lecture, ignore it.\n"
                      "- Do not include material that isn't mentioned in the source material.")

    formatted_prompt = [{"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}]

    total_tokens = count_tokens(system_message) + count_tokens(prompt)
    remaining_tokens = MAX_TOKENS - total_tokens - TOKEN_BUFFER

    if remaining_tokens < 0:
        print(f"Warning! Input text is longer than model gpt-4o can support. Consider trimming input and trying again.")
        print(f"Current length: {total_tokens}, recommended < {MAX_TOKENS - TOKEN_BUFFER}")
        raise ValueError('Input text too long')

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=formatted_prompt,
        max_tokens=remaining_tokens,
        n=1,
        stop=None,
        temperature=temperature)

    return completion.choices[0].message.content.strip()


@handle_api_error
def define_objectives_from_pdf(pdf_file, temperature=1.0):
    text_pages = extract_text_from_pdf(pdf_file)
    all_objectives = []

    system_message = ("You are receiving lecture material for a medical school lesson. Use the following principles when making learning objectives (LO).\n\n"
                      "Material: \"Source Material\"\n\n"
                      "Task: Your task is to analyze the Source Material and condense the information into concise and direct learning objectives. Ensure that learning objectives are clearly written at a level appropriate for medical students while being easily understandable, and adheres to the specified formatting and reference criteria.\n\n"
                      "Formatting Criteria:\n"
                      "- Each LO needs to be succinct, precise, and exactly aligns with the Source Material.\n"
                      "- Each LO must be unique, with minimal overlapping information and stand on their own.\n"
                      "- Limit the word count of each Learning Objective to single sentences with less than 25 words.\n"
                      "- If an abbreviation or acronym is mentioned, include both the expanded name and the abbreviation/acronym in each LO that mentions them.\n\n"
                      "Reference Criteria:\n"
                      "- Each LO must include at most two key terms on pathogenesis, pathophysiology, symptoms, treatments, diagnostics, etc., if mentioned in the Source Material.\n"
                      "- If some material is briefly mentioned as a roadmap and/or stated to be explored in another lecture, ignore it.\n"
                      "- Do not include material that isn't mentioned in the source material.")

    system_token_count = count_tokens(system_message)
    max_chunk_size = MAX_TOKENS - system_token_count - TOKEN_BUFFER

    for page_number, page_text in enumerate(text_pages, start=1):
        print(f"Processing page {page_number}/{len(text_pages)}")  # Debugging print statement

        # Ensure the chunk does not exceed the maximum token limit minus the system message tokens
        if count_tokens(page_text) > max_chunk_size:
            # Truncate the chunk to fit within the token limit
            enc = tiktoken.encoding_for_model("gpt-4o-mini")
            tokens = list(enc.encode(page_text))
            truncated_text = enc.decode(tokens[:max_chunk_size])
        else:
            truncated_text = page_text

        prompt = f"Material: \"{truncated_text}\"\n\nLearning Objectives based on Source Material:"
        generated_text = generate_questions(prompt, temperature=1.0)

        # Extract learning objectives from the response
        objectives = []
        for line in generated_text.split("\n"):
            line_strip = line.strip()
            if line_strip.startswith("1. ") or line_strip.startswith("2. ") or line_strip.startswith("3. "):
                objectives.append(line_strip)

        all_objectives.extend(objectives)

    return all_objectives


@handle_api_error
def generate_embedding(obj, embedding_model="text-embedding-3-small", embedding_encoding="cl100k_base"):
    # Set up the tokenizer
    encoding = tiktoken.get_encoding(embedding_encoding)

    # Generate the tokens and embeddings
    tokens = len(encoding.encode(obj))
    emb = get_embedding(obj, model=embedding_model)

    return tokens, emb


def write_to_csv(csv_writer, output_prefix, objectives):
    n = 0
    for obj in objectives:
        obj_clean = re.sub(r'^\d+\.', '', obj).strip().lstrip('- ')
        remove_words = ['Summary', 'Learning', 'Objective', 'Guiding', 'Additional', 'Question']
        if len([word for word in remove_words if word in obj_clean]) < 2:
            n += 1
            tokens, emb = generate_embedding(obj)
            csv_writer.writerow([output_prefix, obj_clean, tokens, emb])
    print(f"Wrote {n} learning objectives to file for {output_prefix}")


def main(input_path):
    path = Path(input_path)
    output_prefix = path.stem
    output_file = output_prefix + "_learning_objectives.csv"

    if path.is_file():
        pdf_files = [input_path]
    elif path.is_dir():
        pdf_files = list(path.glob('*.pdf'))
    else:
        print("The provided path is not a valid file or directory.")
        sys.exit(1)

    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['name', 'learning_objective', 'tokens', 'emb'])

        for pdf_file in pdf_files:
            print(f"Processing PDF: {pdf_file}")  # Debugging print statement
            objectives = define_objectives_from_pdf(pdf_file)
            tag = Path(pdf_file).stem
            write_to_csv(csv_writer, tag, objectives)


if __name__ == "__main__":
    set_api_key()
    if len(sys.argv) != 2:
        print("Usage: make_learning_objectives.py <pdf_file_or_dir>")
        sys.exit(1)
    path = sys.argv[1]
    main(path)
