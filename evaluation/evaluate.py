# scripts/evaluate.py
import json
import os
import re
import csv
from sentence_transformers import SentenceTransformer, util
from get_html import get_html

# Path to your Q/A JSON data
QA_JSON = "../UMIACSQuestions.json"
RAW_HTML_DIR = "../dataset/raw_html/"
OUTPUT_CSV = "../evaluation_results.csv"

def url_to_filename(url):
    # Extract page name from URL. 
    # Example: https://wiki.umiacs.umd.edu/umiacs/index.php/Network/Troubleshooting/DNS 
    # should become Network_Troubleshooting_DNS.html
    match = re.search(r'index\.php/(.+)$', url)
    if match:
        # Replace slashes with underscores in the page name
        page_name = match.group(1).replace('/', '_')
    else:
        # fallback if not matched
        page_name = os.path.basename(url)
    # Append .html to the page name
    filename = page_name + ".html"
    return filename

def main():
    # Load Q/A data
    with open(QA_JSON, "r") as f:
        qa_data = json.load(f)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results = []
    for page in qa_data:
        url = page["url"]
        qa_pairs = page["Q/A"]
        filename = url_to_filename(url)
        filepath = os.path.join(RAW_HTML_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found for URL {url} -> {filename}")
            continue

        page_text = get_html(filepath)
        page_embedding = model.encode(page_text, convert_to_tensor=True)

        for entry in qa_pairs:
            question = entry["question"]
            answer = entry["answer"]
            answer_embedding = model.encode(answer, convert_to_tensor=True)
            similarity = util.cos_sim(answer_embedding, page_embedding).item()
            # similarity: a rough measure of how well answer text semantically aligns with page content
            results.append([url, question, answer, similarity])
    
    # Write results to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["URL", "Question", "Answer", "Similarity"])
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    main()
