# scripts/visualize.py
# I do not claim the total ownership of this code, help with AI

"""
This script visualizes the similarity scores from a CSV file, showing both individual scores and a rolling 
average.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

RESULTS_CSV = "evaluation_results_chatbot.csv"
OUTPUT_PNG = "evaluation_chart_chatbot.png"
ROLLING_WINDOW = 50  # adjust the rolling window for smoothing as desired

def main():
    questions = []
    similarities = []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["Question"]
            sim = float(row["Similarity"])
            questions.append(q)
            similarities.append(sim)

    # convert similarities to a NumPy array for easy rolling calculation
    sim_arr = np.array(similarities)

    # compute a rolling average to smooth the trend line
    window = np.ones(ROLLING_WINDOW) / ROLLING_WINDOW
    smoothed = np.convolve(sim_arr, window, mode='valid')

    plt.figure(figsize=(10,6))

    x = np.arange(len(similarities))
    # plot bars for individual similarities
    plt.bar(x, similarities, color='blue', alpha=0.5, label='Individual Similarity')

    # plot the rolling average line
    smoothed_x = np.arange(ROLLING_WINDOW - 1, len(similarities))
    plt.plot(smoothed_x, smoothed, color='red', linewidth=2, label='Rolling Average Similarity')

    plt.ylabel("Similarity Score")
    plt.xlabel("Q&A Pairs")
    plt.title("Q/A Answer Similarity to Source Page")

    # remove x-axis ticks/labels for individual questions, just show "Q&A Pairs"
    plt.xticks([])
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    plt.show()

if __name__ == "__main__":
    main()
