# Anki_Tagger


Update 1.1v
 1. Updated the code, as the newest versions of OpenAI no longer support the previous util.embedding. 
 2. Now cycles through multiple PDFs so you only need to run the script once.
  2a. Deposits the completed PDF, along with the .csv files into a seperate folder titles "archive". Jut added this feature incase I wanted to look back on what was the tagged. 
 3. Updated the prompt to select cards with Anking examples. Found that this made the tagging more selective with the cards.
 4. Updated the script to run on gpt-4o mini. Cheaper, faster and better.
 5. Reorganized the scripts so you only need to run "tag_cards.py" to make objectives, select and tag cards.
 6. This is just a personal thing, but I made "tag_cards.py" copy and paste the anki.apkg, as to not contaminate the orginal anki deck. I chose to add this feature because I also tag the cards for my M1 class.
 7. Waits after all the PDFs are completed to tag the deck. This feature was to minimize overlapping cards between lecutures.

----------
This is a project I started as a fourth year medical student. Anki was a huge part of my medical education, and this felt like a productive way to give back for everything others have done.
 
Big picture, this set of scripts will parse a lecture guide and identify the most relevant anki cards within a premade Anki deck. 

While there are some fantastic deck’s out there to support medical education, aligning the content of these decks with preclinical curriculum is a persistent challenge and source of anxiety for students. This project aims to alleviate that stress by selecting the best cards for each lecture to study alongside their preclinical curriculum. Using the OpenAI API, I was able to quickly tag over 200 lectures to cover the entirety of M1 and M2 with minimal human intervention. While it is not the perfect solution, hopefully others can use and improve upon the code and make their own class tags!

#Workflow
1. Create a personal OpenAI account, and obtain an API key.
2. In anki, export the deck you wish to tag as an anki_deck.apkg.
3. In anki, export the deck using the Notes as plain text funcion, and select to include a unique identifier: anki.txt
4. python embed_anki_deck.py <anki.txt>
Returns: anki_embeddings.csv
This will create the embeddings of your deck: These are required for a first pass crude search of your deck to minimize API costs.
5. python make_learning_objectives.py <learning_guide.pdf> or <folder_of_pdfs>
Returns: anki_learning_objectives.csv
Create a list of summary learning objectives, the filename of the pdf will be the tag for the learning objective. Generally one lecture guide results in 10-30 questions.
6. python select_cards.py <deck_embeding> <learning_objectives> 
Returns: anki_cards.csv
This will create a list of cards from your deck scoring them on their relevance to each learning objective.
7. python tag_deck.py <anki_cards.csv> <anki_deck.apkg>
Will tag the deck, and return the original deck apkg file.
8. Import into Anki and enjoy!

zachalmers

