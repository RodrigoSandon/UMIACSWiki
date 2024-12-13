# I do not claim the total ownership of this code, help with AI
"""
This code compares and visualizes the performance of two different systems (a chatbot and a baseline)
based on cosine similarity scores for both.
"""
import pandas as pd
import matplotlib.pyplot as plt

# load the CSV files
file1 = 'evaluation_results_chatbot.csv'
file2 = '../evaluation_results.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# merge the dataframes on 'URL' and 'Question', keeping only matching rows
merged_df = pd.merge(df1, df2, on=['URL', 'Question'])

# calculate rolling averages and variances for both similarity scores
window_size = 5  # adjust the window size as needed

# for similarity from file1 (chatbot)
merged_df['Rolling_Avg_1'] = merged_df['Similarity_x'].rolling(window=window_size).mean()
merged_df['Rolling_Var_1'] = merged_df['Similarity_x'].rolling(window=window_size).var()

# for similarity from file2 (baseline)
merged_df['Rolling_Avg_2'] = merged_df['Similarity_y'].rolling(window=window_size).mean()
merged_df['Rolling_Var_2'] = merged_df['Similarity_y'].rolling(window=window_size).var()

plt.figure(figsize=(12, 6))

plt.errorbar(merged_df.index, 
            merged_df['Rolling_Avg_1'], 
            yerr=merged_df['Rolling_Var_1'], 
            fmt='-', 
            label='Chatbot Similarity',
            color='blue',
            alpha=0.7)

plt.errorbar(merged_df.index, 
            merged_df['Rolling_Avg_2'], 
            yerr=merged_df['Rolling_Var_2'], 
            fmt='-', 
            label='Baseline Similarity',
            color='red',
            alpha=0.7)

plt.xlabel('Index')
plt.ylabel('Similarity Score')
plt.title('Rolling Average of Similarity Scores with Variance')
plt.legend()
plt.grid(True)
plt.savefig('rolling_average_similarity_comparison.png')
plt.show()
