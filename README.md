# ViCom CollabMap

ViCom CollabMap is a data science project designed to visually present existing collaborations within the ViCom research group and uncover potential new connections. This project processes researcher data, extracts expertise, and generates potential collaborations, all presented in an interactive web-based map.

Sometimes, you want to know who else is quietly exploring **eye-tracking** or deep into **Bayesian stats**—without turning into Sherlock Holmes. **ViCom CollabMap** is here to save the day, letting you see where collaborations already exist and where they might blossom if only you knew the right people.

*(P.S. This entire project was built in just three weeks. So please forgive the occasional quirk or feature that’s still “under construction.”)*

---

## Table of Contents
- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Detailed Notebook Workflow](#detailed-notebook-workflow)
  - [02.Extract_Publication_data.ipynb](#02extract_publication_dataipynb)
  - [03.Clean_Publication_data.ipynb](#03clean_publication_dataipynb)
  - [04.Data_processing_with_OpenAI.ipynb](#04dataprocessingwithopenaiipynb)
  - [05.Clean_OpenAI_data.ipynb](#05clean_openai_dataipynb)
  - [06.Similarity.ipynb](#06similarityipynb)
  - [app.py](#apppy)
- [How to Use Each Notebook](#how-to-use-each-notebook)
- [Example Outputs](#example-outputs)
- [Error Handling](#error-handling)
- [Improvement Ideas](#improvement-ideas)
- [Files in the Repository](#files-in-the-repository)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Technologies Used](#technologies-used)
- [Final Thoughts](#final-thoughts)

---

## Overview

This project was completed over the span of **3 weeks**, primarily focusing on data processing and a bit of web-based visualization. It aims to:
- Collect researcher info and publication data (via **OpenAlex**).
- Generate meaningful keywords with the help of **OpenAI**.
- Identify potential synergies between researchers.
- Show existing and potential collaborations on a lovely interactive map, courtesy of **Streamlit** + **Folium**.

---

## Project Workflow

1. **Data Gathering**  
   Researchers’ details (names, affiliations, bios, and geolocation) were scraped from the ViCom website. Then, using **OpenAlex**, 1,264 recent publications were fetched.

2. **Data Preprocessing & Cleaning**  
   Publications were cleaned for duplicates, missing titles, etc., and merged with geolocation data for each researcher.

3. **Expertise Extraction**  
   **OpenAI** was then employed to read through abstracts and produce relevant themes and expert keywords. Who doesn’t want a robot telling us our life’s work in bullet points?

4. **Similarity Computation**  
   We tally up how many keywords each pair of researchers shares—like counting how many matching socks you’ve got in the laundry.

5. **Web App Visualization**  
   Finally, we created a **Streamlit** + **Folium** web app to:
   - Show a map of where each researcher is located.
   - Draw lines for existing collaborations (blue/green lines).
   - Visualize potential collaborations via purple arcs.

---

## Detailed Notebook Workflow

### 02.Extract_Publication_data.ipynb
1. **Input:** A CSV of researchers (names, affiliations).  
2. **Process:**  
   - Queries **OpenAlex** for each name.
   - Fetches publication data (title, abstract, authors, year) from the last 5 years.
   - Reconstruct abstracts from OpenAlex’s “inverted index” (like puzzle-solving with words).
3. **Output:** A CSV (`03.ViCom_Publications_OpenAlex.csv`) with all the extracted data.

### 03.Clean_Publication_data.ipynb
1. **Input:** The raw file from the previous notebook.  
2. **Process:**  
   - Fills in missing columns (title, abstract).
   - Removes duplicates and lumps data into a single “Text” field, also merges with researcher bios.
3. **Output:** A cleaned CSV (`03.ViCom_Publications_OpenAlex_Cleaned.csv`) and a processed file (`06. Processed_Researcher_Data.csv`).

### 04.Data_processing_with_OpenAI.ipynb
1. **Input:** The merged file of abstracts + bios.  
2. **Process:**  
   - Splits each researcher’s text into smaller chunks (avoiding token-limit drama).
   - Calls **OpenAI** to extract keywords: “Themes” vs. “Expertise.”
   - Combines them back into a single row per researcher.
3. **Output:** A CSV (`08.researchers_with_themes_expertise_openai.csv`) listing each researcher’s themes and expertise.

### 05.Clean_OpenAI_data.ipynb
1. **Input:** The raw OpenAI keyword data.  
2. **Process:**  
   - Applies synonyms (merging “sign language” with “sign language,” etc.).
   - Uses fuzzy matching to remove near-duplicates.
   - Drops generic or irrelevant terms (“education,” “research focus,” etc.).
3. **Output:** A tidy CSV (`08.researchers_with_themes_expertise_cleaned.csv`) with standardized keywords.

### 06.Similarity.ipynb
1. **Input:** Cleaned expertise data.  
2. **Process:**  
   - For each pair of researchers, count overlapping keywords (Themes + Expertise).
   - Saves the synergy count and the list of matching keywords (like “EEG,” “lexical semantics,” etc.).
3. **Output:** A CSV (`09.potential_collaborations.csv`) listing synergy scores for every possible pair.

### app.py
1. **Input:** Multiple CSVs (researcher info, collaborations, synergy scores).  
2. **Process:**  
   - **Streamlit** app with optional password protection.
   - Builds an interactive **Folium** map with researcher markers, lines for existing collaborations, and arcs for potential ones.
   - Lets you filter by project type or synergy threshold.
3. **Output:** Shiny interactive map and tables of who’s working (or should be working) with whom.

---

## How to Use Each Notebook

1. **02.Extract_Publication_data.ipynb**  
   - Update the file paths to point at your researcher CSV.
   - Make sure your internet is up—OpenAlex can’t be reached by carrier pigeon.
   - Run all cells to get a CSV of each researcher’s publications.

2. **03.Clean_Publication_data.ipynb**  
   - Point it to the CSV from the previous step.
   - Run all cells to produce cleaned data and a combined “Text” field.

3. **04.Data_processing_with_OpenAI.ipynb**  
   - Insert your **OpenAI API key** (replace `"xxx"`).
   - Adjust `MAX_CHARS_PER_CHUNK` if your texts are huge.
   - Run all cells; wait for GPT to parse your data.

4. **05.Clean_OpenAI_data.ipynb**  
   - Load the new CSV from the OpenAI pipeline.
   - Click “Run” and watch it unify synonyms and remove fluff.

5. **06.Similarity.ipynb**  
   - Uses the final “cleaned” data.
   - Run it to get synergy scores for each pair of researchers.

6. **app.py**  
   - Check you have **Streamlit** and **Folium** installed.
   - Add `.streamlit/secrets.toml` with your chosen password.
   - `streamlit run app.py` -> Follow the link in your terminal -> Enjoy the map!

---

## Error Handling

### API Rate Limits
If OpenAI or OpenAlex complains about too many requests, add a small time delay in the loops (e.g., `time.sleep(2)`).

### Missing Geolocation
Double-check the `Latitude` and `Longitude` columns in `01_participants_with_geo.csv`. A blank lat/long means no map marker.

### JSON Decoding Woes
Sometimes GPT’s output isn’t perfect JSON. This notebook tries to handle that, but you may need to re-run or modify the prompt.

### No Publications Found
Some researchers may genuinely not have entries in OpenAlex or have incomplete info. In those cases, the script logs “No publications found.”

---

## Improvement Ideas

### Smarter Keywords
Right now, we rely on GPT for keywords. You could integrate domain-specific dictionaries or trained classifiers for more accuracy.

### Extended Publications
For older or specialized data, consider using other APIs (e.g., PubMed, CrossRef) to widen the publication range.

### Add More Filters
Let users filter by “department,” “method,” or “region.” Not everyone enjoys endless scrolling.

### Automated Pipelines
Use GitLab CI or GitHub Actions to refresh the data nightly so you’re always up-to-date with the newest preprints.

### Better UI
If you have a knack for design, feel free to refactor the interface for a more polished (or simpler) look.

---

## Files in the Repository

- **02.Extract_Publication_data.ipynb**  
  Pulls data from OpenAlex for each researcher.

- **03.Clean_Publication_data.ipynb**  
  Fixes missing values, merges data fields, and aggregates text.

- **04.Data_processing_with_OpenAI.ipynb**  
  Feeds abstracts to OpenAI for keyword extraction.

- **05.Clean_OpenAI_data.ipynb**  
  Removes duplicates, merges synonyms, cleans up categories.

- **06.Similarity.ipynb**  
  Calculates synergy scores for researcher pairs.

- **app.py**  
  The interactive Streamlit app for exploring and filtering collaborations.

---

## How to Run the Project Locally

### **Clone the Repo**
```bash
git clone <repository_url>
cd ViCom

### **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


### Install Requirements
```bash
pip install -r requirements.txt
```

### Install Requirements
```bash
password = "whatever_password_you_want"
```

### Run the App
```bash
streamlit run app.py
```
Open Your Browser
Visit http://localhost:8501 and log in with your password.

## Technologies Used

- **Python**
- **Streamlit**
- **Folium**
- **OpenAI API**
- **OpenAlex API**
- **Pandas & NumPy**

*(Who doesn’t love a data pipeline with a dash of AI?)*

---

## Final Thoughts

This project was a solo sprint, so if you spot anything bizarre, please bear with me. **ViCom CollabMap** is here to help you discover your next big research partnership—or at least give you a fun excuse to reach out to that colleague you’ve been curious about. Enjoy!




