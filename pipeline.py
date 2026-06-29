import os
import requests
import time

# Path setup
CORPUS_PATH = "01_korpus_raw"
RESULTS_PATH = "03_ergebnisse"
OPENAI_BATCH_SIZE = 4  

# Create the results directory if it does not exist yet
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load API keys from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_stories_list(folder_path):
    """Reads all .txt files and returns a list of tuples: (filename, content)."""
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' was not found.")
        return []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
    return [(filename, open(os.path.join(folder_path, filename), "r", encoding="utf-8").read()) for filename in files]

def get_triangulation_prompt(stories_subset, info_text=""):
    """Generates the literary prompt for a given selection of stories."""
    corpus_text = ""
    for filename, content in stories_subset:
        corpus_text += f"\n\n--- START OF STORY FILE: {filename} ---\n{content}\n--- END OF STORY FILE: {filename} ---\n"
        
    return f"""
    You are an expert in German Studies (Germanistik) and Digital Humanities. 
    Analyze the following collection of short stories by Otto Anthes from the book 'Lübeck du seltsam schöne Stadt'.
    {info_text}
    
    Please provide a structured comparative analysis. For EACH individual story, include:
    1. PLOT SUMMARY: A concise summary of the narrative (max. 150 words per story).
    2. SPATIAL REPRESENTATION OF LÜBECK: How is the city of Lübeck represented (topographically, atmospherically, emotionally)? Which specific locations are mentioned?
    3. THEMES & MOTIFS: Which central themes (e.g., nostalgia, tradition, bourgeoisie, art vs. commerce) can be identified?
    
    Here is the text corpus of the stories:
    {corpus_text}
    """

def analyze_with_gemini(prompt_content):
    """Sends the entire corpus to Gemini in ONE single call (Gemini supports huge outputs)."""
    if not GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt_content}]}]}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

def analyze_with_openai_batch(prompt_content):
    """Sends a smaller batch prompt to OpenAI (gpt-4o-mini)."""
    if not OPENAI_API_KEY:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt_content}]
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None

if __name__ == "__main__":
    print("Starting Hybrid Production Pipeline...")
    
    all_stories = get_stories_list(CORPUS_PATH)
    if not all_stories:
        print("No stories found to process.")
    else:
        print(f"Loaded {len(all_stories)} stories locally.")
        
        # --- 1. GEMINI:
        if GEMINI_API_KEY:
            print("\n--- Running Gemini (All-In-One Mode) ---")
            full_prompt = get_triangulation_prompt(all_stories, "(Analyzing all 24 stories together)")
            print("Sending all 24 stories to Gemini...")
            gemini_analysis = analyze_with_gemini(full_prompt)
            if gemini_analysis:
                with open(os.path.join(RESULTS_PATH, "gesamtanalyse_gemini.txt"), "w", encoding="utf-8") as f:
                    f.write(gemini_analysis)
                print("SUCCESS: Saved full Gemini analysis.")
        
        # --- 2. OPENAI:
        if OPENAI_API_KEY:
            print("\n--- Running OpenAI (Smart Batch Mode) ---")
            full_openai_analysis = "=== COMPREHENSIVE OPENAI BATCH ANALYSIS ===\n\n"
            
            total_stories = len(all_stories)
            for i in range(0, total_stories, OPENAI_BATCH_SIZE):
                batch = all_stories[i:i + OPENAI_BATCH_SIZE]
                batch_num = (i // OPENAI_BATCH_SIZE) + 1
                print(f"Processing OpenAI Batch {batch_num} (Stories {i+1} to {min(i+OPENAI_BATCH_SIZE, total_stories)})...")
                
                openai_prompt = get_triangulation_prompt(batch, f"(Batch {batch_num} of {int(total_stories/OPENAI_BATCH_SIZE)})")
                batch_result = analyze_with_openai_batch(openai_prompt)
                
                if batch_result:
                    full_openai_analysis += f"\n\n{'#'*40}\nBATCH {batch_num} RESULTS\n{'#'*40}\n\n"
                    full_openai_analysis += batch_result
                

                if i + OPENAI_BATCH_SIZE < total_stories:
                    time.sleep(2)
            
       
            with open(os.path.join(RESULTS_PATH, "gesamtanalyse_openai.txt"), "w", encoding="utf-8") as f:
                f.write(full_openai_analysis)
            print("SUCCESS: Saved complete pieced-together OpenAI analysis.")

    print("\nPipeline successfully completed. Check your '03_ergebnisse' folder!")