# Comparative Literary Analysis Pipeline: Otto Anthes – 'Lübeck du seltsam schöne Stadt'

This repository contains a hybrid production pipeline developed for the course 'GenAI for Humanists'. It leverages state-of-the-art Large Language Models (LLMs) to perform a structured, comparative digital humanities analysis of a short story corpus.

The pipeline specifically analyzes a collection of 24 short stories by Otto Anthes from his book 'Lübeck du seltsam schöne Stadt', evaluating spatial representations, plot summaries, and central motifs.

---

## Features & Hybrid Methodology

The pipeline uses a **dual-model approach** to cross-examine and triangulate literary insights:

1. **Gemini 2.5 Flash (All-In-One Mode):** Utilizes Gemini's massive context window to process all 24 stories simultaneously in a single API call, providing a holistic comparative overview.
2. **OpenAI gpt-4o-mini (Smart Batch Mode):** Processes the corpus sequentially in smaller, optimized chunks (batches of 4 stories) to ensure highly detailed, granular tracking of individual narratives without losing precision.

### Analytical Focus
For every story in the corpus, the models extract:
* **Plot Summary:** Concise narrative abstract (max. 150 words).
* **Spatial Representation of *Lübeck*:** Topographical, atmospheric, and emotional depiction of specific locations.
* **Themes & Motifs:** Key literary subjects (e.g., nostalgia, tradition, bourgeoisie, art vs. commerce).

---

## Repository Structure

```text
├── 01_korpus_raw/          # Input directory containing the text files (.txt) of the stories
├── 03_ergebnisse/          # Output directory where the final analyses are stored
├── pipeline.py             # The main Python script execution file
├── .gitignore              # Standard git exclusion file (hides local keys & cache)
└── requirements.txt        # Python package dependencies


🛠️ Installation & Setup

1. Clone the Repository
Bash
git clone [https://github.com/n8r7t7wrdn-gif/Nik-Schumacher-Final-Project-GenAI.git](https://github.com/n8r7t7wrdn-gif/Nik-Schumacher-Final-Project-GenAI.git)
cd Nik-Schumacher-Final-Project-GenAI

2. Install Dependencies
Make sure you have Python installed, then run:
Bash
pip install -r requirements.txt

3. Set Up API Keys (Environment Variables)
The pipeline securely fetches API credentials from your system environment variables. You need to set them before running the script.
On macOS / Linux (Terminal):
Bash
export GEMINI_API_KEY="your-gemini-api-key-here"
export OPENAI_API_KEY="your-openai-api-key-here"
On Windows (Command Prompt):
DOS
set GEMINI_API_KEY="your-gemini-api-key-here"
set OPENAI_API_KEY="your-openai-api-key-here"


💻 Usage
Place your text files inside the 01_korpus_raw/ folder (formatted as .txt, sorted numerically or alphabetically).
Execute the hybrid pipeline:
Bash
python pipeline.py
Check the 03_ergebnisse/ folder for the generated analytical reports:
gesamtanalyse_gemini.txt (Global overview)
gesamtanalyse_openai.txt (Detailed batch results)

