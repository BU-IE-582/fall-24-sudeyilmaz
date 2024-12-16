#!/usr/bin/env python
# coding: utf-8

# # ***Sude Yılmaz***
# 
# 
# # ***IE 582 Homework2***
# 
# 
# ## [Link to my GitHub Repository](https://bu-ie-582.github.io/fall-24-sudeyilmaz/).
# 

# In[50]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import spsolve  
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# # Data Preprocessing

# In[2]:


match_data_raw = pd.read_csv("match_data.csv")

# remove instances where “suspended” values are true or “stopped” values are true
match_data_filtered = match_data_raw[(match_data_raw['suspended'] == False) & (match_data_raw['stopped'] == False)].copy()

# remove instances where "1", "X", "2" values are Na
match_data = match_data_filtered.dropna(subset=["1", "X", "2"])

# remove rows where "1", "X", "2" values are empty
match_data = match_data[(match_data["1"] != "") & (match_data["X"] != "") & (match_data["2"] != "")]


# In[3]:


match_data.head()


# In[5]:


# Filter matches where "name" contains "Galatasaray"
matches_with_home_goals = sorted(
    list(
        set(
            match_data[match_data["name"].str.contains("Galatasaray", case=False, na=False)]["fixture_id"]
        )
    )
)

for selected_match in matches_with_home_goals[:100]: # show some of the games
    selected_match_data = match_data[match_data["fixture_id"] == selected_match].reset_index(drop=True)

    plt.plot(np.log(selected_match_data["1"]))
    plt.plot(np.log(selected_match_data["X"]))
    plt.plot(np.log(selected_match_data["2"]))

    plt.title("Final Result Odds\n" + selected_match_data["name"][0] + " " + selected_match_data["final_score"][0])

    plt.vlines(np.where(selected_match_data["Score Change - home"] == 1),0,np.log(selected_match_data[["1","2","X"]].max().max()),color="red",linestyles="dashed")
    plt.vlines(np.where(selected_match_data["Score Change - away"] == 1),0,np.log(selected_match_data[["1","2","X"]].max().max()),color="black",linestyles="dashed")
    plt.vlines(np.where(selected_match_data["Score Change - home"] == -1),0,np.log(selected_match_data[["1","2","X"]].max().max()),color="salmon",linestyles="dotted")
    plt.vlines(np.where(selected_match_data["Score Change - away"] == -1),0,np.log(selected_match_data[["1","2","X"]].max().max()),color="gray",linestyles="dotted")

    plt.legend(["Home Log Odd","Draw Log Odd","Away Log Odd","Home Goal","Away Goal","Home Goal Cancel","Away Goal Cancel"])
    plt.show()


# # Task 1

# ### Q1

# Calculating:
# - P(home win) = 1/"1"
# - P(tie) = 1/"X"
# - P(away win) = 1/"2"

# In[4]:


match_data['prob_home'] = 1 / match_data['1'].astype(float)  
match_data['prob_draw'] = 1 / match_data['X'].astype(float)
match_data['prob_away'] = 1 / match_data['2'].astype(float)


# ## Visualizing Final Result Probabilities for Matches Involving "Reims" Based on Calculations Above

# In[5]:


selected_matches = sorted(
    list(
        set(
            match_data[match_data["name"].str.contains("Reims", case=False, na=False)]["fixture_id"]
        )
    )
)

for selected_match in selected_matches:
    selected_match_data = match_data[match_data["fixture_id"] == selected_match].reset_index(drop=True)

    plt.plot(selected_match_data["prob_home"])
    plt.plot(selected_match_data["prob_draw"])
    plt.plot(selected_match_data["prob_away"])

    plt.title("Final Result Probabilities\n" + selected_match_data["name"][0] + " " + selected_match_data["final_score"][0])

    plt.vlines(np.where(selected_match_data["Score Change - home"] == 1),0,selected_match_data[["prob_home","prob_draw","prob_away"]].max().max(),color="red",linestyles="dashed")
    plt.vlines(np.where(selected_match_data["Score Change - away"] == 1),0,selected_match_data[["prob_home","prob_draw","prob_away"]].max().max(),color="black",linestyles="dashed")
    plt.vlines(np.where(selected_match_data["Score Change - home"] == -1),0,selected_match_data[["prob_home","prob_draw","prob_away"]].max().max(),color="salmon",linestyles="dotted")
    plt.vlines(np.where(selected_match_data["Score Change - away"] == -1),0,selected_match_data[["prob_home","prob_draw","prob_away"]].max().max(),color="gray",linestyles="dotted")

    plt.legend(["P(HomeWin)","P(Tie)","P(AwayWin)","Home Goal","Away Goal","Home Goal Cancel","Away Goal Cancel"])
    plt.show()


# ### Q2

# Normalizing Probabilities by dividing them to sum of final probabilities:
# - P_norm(home win) = (1/"1") / ( (1/"1") + (1/"X") + (1/"2") )
# - P_norm(tie) = (1/"X") / ( (1/"1") + (1/"X") + (1/"2") )
# - P_norm(away win) = (1/"2") / ( (1/"1") + (1/"X") + (1/"2") )

# In[6]:


match_data['norm_factor'] = (1 / (match_data['prob_home'] + match_data['prob_draw'] + match_data['prob_away'])).astype(float)
match_data['prob_home_norm'] = (match_data['prob_home']*match_data['norm_factor']).astype(float)
match_data['prob_draw_norm'] = (match_data['prob_draw']*match_data['norm_factor']).astype(float)
match_data['prob_away_norm'] = (match_data['prob_away']*match_data['norm_factor']).astype(float)


# ## Visualizing Final Result Normalized Probabilities for Matches Involving "Crystal Palace" Based on Calculations Above

# In[7]:


selected_matches = sorted(
    list(
        set(
            match_data[match_data["name"].str.contains("Crystal Palace", case=False, na=False)]["fixture_id"]
        )
    )
)

for selected_match in selected_matches:
    selected_match_data = match_data[match_data["fixture_id"] == selected_match].reset_index(drop=True)

    plt.plot(selected_match_data["prob_home_norm"])
    plt.plot(selected_match_data["prob_draw_norm"])
    plt.plot(selected_match_data["prob_away_norm"])

    plt.title("Final Result Odds\n" + selected_match_data["name"][0] + " " + selected_match_data["final_score"][0])

    plt.vlines(np.where(selected_match_data["Score Change - home"] == 1),0,selected_match_data[["prob_home_norm","prob_draw_norm","prob_away_norm"]].max().max(),color="red",linestyles="dashed")
    plt.vlines(np.where(selected_match_data["Score Change - away"] == 1),0,selected_match_data[["prob_home_norm","prob_draw_norm","prob_away_norm"]].max().max(),color="black",linestyles="dashed")
    plt.vlines(np.where(selected_match_data["Score Change - home"] == -1),0,selected_match_data[["prob_home_norm","prob_draw_norm","prob_away_norm"]].max().max(),color="salmon",linestyles="dotted")
    plt.vlines(np.where(selected_match_data["Score Change - away"] == -1),0,selected_match_data[["prob_home_norm","prob_draw_norm","prob_away_norm"]].max().max(),color="gray",linestyles="dotted")

    plt.legend(["Normalized P(HomeWin)","Normalized P(Draw)","Normalized P(Away)","Home Goal","Away Goal","Home Goal Cancel","Away Goal Cancel"])
    plt.show()


# ### Q3

# ### The function plotter1:
# It takes two inputs: the half-time and a flag indicating whether to use normalized or non-normalized probabilities for plotting. It performs the following tasks:
# 
# - Calculates P(home) - P(away) based on the given half-time and normalization criteria.
# - Computes the total time in minutes to enhance the visualization of the graph.
# - Plots the graph with P(home) - P(away) on the x-axis and P(tie) on the y-axis.
# - Color-maps the graph according to minute information, providing visual insights into the relationship between the graph and time.

# In[8]:


def plotter1(df, half='1st-half', normalized=1, ax=None):
    # Filter for halftime and create a copy
    filtered_data = df[df['halftime'] == half].copy()

    # Calculate x and y
    if normalized == 0:
        normalized_str = ''
        filtered_data['x'] = filtered_data['prob_home'] - filtered_data['prob_away']
        filtered_data['y'] = filtered_data['prob_draw']
    else:
        normalized_str = ', normalized'
        filtered_data['x'] = filtered_data['prob_home_norm'] - filtered_data['prob_away_norm']
        filtered_data['y'] = filtered_data['prob_draw_norm']

    # Calculate total time in minutes to use as a color mapping
    if half == '1st-half':
        filtered_data['time_in_minutes'] = filtered_data['minute'] + filtered_data['second'] / 60
        colormap_vmin = 0
        colormap_vmax = 45
    else:
        filtered_data['time_in_minutes'] = 45 + filtered_data['minute'] + filtered_data['second'] / 60
        colormap_vmin = 45
        colormap_vmax = 90
    
    # Plot
    if ax is None:
        ax = plt.gca()  # Use the current axis if not provided

    scatter = ax.scatter(
        filtered_data['x'], 
        filtered_data['y'], 
        c=filtered_data['time_in_minutes'], 
        cmap='viridis_r', 
        s=3,  
        alpha=0.7,
        vmin=colormap_vmin,  
        vmax=colormap_vmax
    )

    # Add a colorbar to show the mapping
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (minutes)')

    # Add labels and title
    ax.set_title(f'Scatter Plot of [P(home win) - P(away win)] vs P(tie)\n{half}{normalized_str}')
    ax.set_xlabel('P(home win) - P(away win)')
    ax.set_ylabel('P(tie)')

    # Show grid on the specific subplot
    ax.grid(True, linestyle='--', alpha=0.7)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot for 1st-half normalized=0
plotter1(match_data, half='1st-half', normalized=0, ax=axes[0, 0])

# Plot for 1st-half normalized=1
plotter1(match_data, half='1st-half', normalized=1, ax=axes[1, 0])

# Plot for 2nd-half normalized=0
plotter1(match_data, half='2nd-half', normalized=0, ax=axes[0, 1])

# Plot for 2nd-half normalized=1
plotter1(match_data, half='2nd-half', normalized=1, ax=axes[1, 1])

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the grid of plots
plt.show()


# ### The above graphes show that:
# - One common observation is that as P(home win) - P(away win) approaches zero, P(tie) approaches to higher values even to 1 for 2nd-half graphs. This is expected because when there is minimal difference between the winning probabilities of the home and away teams, a tie becomes the most likely outcome.
# 
# - As the match progresses toward the end (darker-colored regions), it becomes easier to make more certain predictions. For example, in the "Scatter Plot of P(home win) - P(away win) vs. P(tie), 2nd-half, normalized" graph, when P(home win) - P(away win) = 0, the green regions (earlier time of the 2nd-half) show a P(tie) value of around 0.3. In contrast, the purple regions (later times of the 2nd-half) indicate a P(tie) value of approximately 1. This suggests that when there is more time remaining in the game, a tie cannot be confidently predicted even if P(home win) - P(away win) = 0. However, as the match nears its conclusion, P(tie) becomes certain at 1 in such scenarios.
# 
# - Additionally, in both 1st-half graphs, the highest P(tie) value observed is around 0.5. This is because, during the 1st half, it is harder to predict a tie without accounting for the events of the 2nd half.

# ### The function plotter2:
# It takes two inputs: the half-time and a flag indicating whether to use normalized or non-normalized probabilities for plotting.  
# 
# The key difference compared to the previous plots is that these graphs are color-mapped based on the actual outcomes of the games.  
# 
# - Regions where the game ended in a tie are colored in green. 
# - Regions where the game did not end in a tie are shown in gray. 
# 
# This approach provides a clear visual distinction between tie and non-tie outcomes.

# In[10]:


def plotter2(df, half='1st-half', normalized=1, ax=None):
    # Filter for halftime and create a copy
    filtered_data = df[df['halftime'] == half].copy()

    # Calculate x and y
    if normalized == 0:
        normalized_str = ''
        filtered_data['x'] = filtered_data['prob_home'] - filtered_data['prob_away']
        filtered_data['y'] = filtered_data['prob_draw']
    else:
        normalized_str = ', normalized'
        filtered_data['x'] = filtered_data['prob_home_norm'] - filtered_data['prob_away_norm']
        filtered_data['y'] = filtered_data['prob_draw_norm']

    # Calculate total time in minutes to use as a color mapping
    if half == '1st-half':
        filtered_data['time_in_minutes'] = filtered_data['minute'] + filtered_data['second'] / 60
        colormap_vmin = 0
        colormap_vmax = 45
    else:
        filtered_data['time_in_minutes'] = 45 + filtered_data['minute'] + filtered_data['second'] / 60
        colormap_vmin = 45
        colormap_vmax = 90
    
    # Create a color map based on 'result' column
    filtered_data['color'] = np.where(filtered_data['result'] == 'X', 'green', 'grey')

    # Plot
    if ax is None:
        ax = plt.gca()  # Use the current axis if not provided

    scatter = ax.scatter(
        filtered_data['x'], 
        filtered_data['y'], 
        c=filtered_data['color'], 
        s=6,  
        alpha=0.05
    )

    # Add labels and title
    ax.set_title(f'Scatter Plot of [P(home win) - P(away win)] vs P(tie)\n{half}{normalized_str}')
    ax.set_xlabel('P(home win) - P(away win)')
    ax.set_ylabel('P(tie)')

    # Show grid on the specific subplot
    ax.grid(True, linestyle='--', alpha=0.7)

    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='X'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Not X')]
    ax.legend(handles=handles, loc='upper right')

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot for 1st-half normalized=0
plotter2(match_data, half='1st-half', normalized=0, ax=axes[0, 0])

# Plot for 1st-half normalized=1
plotter2(match_data, half='1st-half', normalized=1, ax=axes[1, 0])

# Plot for 2nd-half normalized=0
plotter2(match_data, half='2nd-half', normalized=0, ax=axes[0, 1])

# Plot for 2nd-half normalized=1
plotter2(match_data, half='2nd-half', normalized=1, ax=axes[1, 1])

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the grid of plots
plt.show()


# ### The above graphes show that:
# 
# Here’s a refined version of your sentences:
# 
# - The regions where the green color becomes denser are more noticeable in the 2nd-half graph, particularly when P(tie) exceeds 0.6.  
# - For edge cases with a significant difference between P(home win) and P(away win), the gray color is more likely to appear.  
# 
# 

# ## Determining the Bins:
# 
# Below code;
# - first determines the bins
# - calculates the number of games ended as "tie" in each bin
# - calculates the number of games ended as "tie" divided by total games in each bin as "estimated P(tie)"
# - gets the mean and median of "prob_draw" probabilities in each bin
# 

# In[11]:


# Define bins for discretizing P(home win) - P(away win)
bins = np.linspace(-1, 1, 11)  
labels = [f"({bins[i]:.1f}, {bins[i+1]:.1f}]" for i in range(len(bins) - 1)]

# Add a column for the discretized bins
match_data['bin'] = pd.cut(
    match_data['prob_home'] - match_data['prob_away'], 
    bins=bins, 
    labels=labels, 
    include_lowest=True
)

# Calculate the number of games ended as "Draw" and total games in each bin
bin_stats = match_data.groupby('bin').agg(
    draws=('result', lambda x: (x == 'X').sum()),  # Count of "Draw" games
    total_games=('result', 'size'),  # Total games in the bin
    median_prob_draw=('prob_draw', 'median'),  # Median of prob_draw in each bin
    mean_prob_draw=('prob_draw', 'mean')  # Mean of prob_draw in each bin
).reset_index()

# Add a column for the estimated probability of draws
bin_stats['Estimated P(tie)'] = bin_stats['draws'] / bin_stats['total_games']

# Round the 'P(draw)' to 4 decimals
bin_stats['Estimated P(tie)'] = (bin_stats['Estimated P(tie)']).round(4)

# Round the 'mean_prob_draw' and 'median_prob_draw' to 4 decimals
bin_stats['mean_prob_draw'] = (bin_stats['mean_prob_draw']).round(4)
bin_stats['median_prob_draw'] = (bin_stats['median_prob_draw']).round(4)

# Display the result in one line
print(bin_stats.to_string(index=False))


# ### The above results show that: 
# 
# - most of the time probabilities proposed by the bookmarker are very close to the probabilities calcualted from the sample.
# - for the intervals where sample's probability is larger than the bookmarker's probability which are:
#     - (-1.0, -0.8] 
#     - (-0.6, -0.4] 
#     - (0.0, 0.2] 
#     - (0.2, 0.4] 
#     there is a probabiltiy of making money by betting "Draw" in the long run.
#  

# In[19]:


bin_stats_name = pd.DataFrame({
    'Bin': ["(-1.0, -0.8]", "(-0.8, -0.6]", "(-0.6, -0.4]", "(-0.4, -0.2]", 
            "(-0.2, 0.0]", "(0.0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", 
            "(0.6, 0.8]", "(0.8, 1.0]"]})


x = np.arange(len(bin_stats_name['Bin']))
probabilities = np.array(bin_stats['Estimated P(tie)'])
mean_probs = np.array(bin_stats['mean_prob_draw'])

plt.figure(figsize=(10, 6))
plt.bar(x, probabilities, color='skyblue', alpha=0.8, edgecolor='black', label="Estimated P(tie)")

# Adding the line plot
plt.plot(x, mean_probs, color='red', marker='o', linestyle='-', label="Mean Prob Draw")

# Customizing the plot
plt.xticks(x, bin_stats_name['Bin'], rotation=45, ha="right")
plt.xlabel("Bins")
plt.ylabel("Probability")
plt.title("Probability Distribution Across Bins")
plt.legend()
plt.tight_layout()

plt.show()



# From the above histogram, it can be observed that there is a slight bias towards the left. This indicates that (p(home win)- p(away win)) being greater than 0 is more likely, as the home team is expected to have a higher probability of winning than the away team.

# # Task 2

# 1) Filter the matches with Na values for the columns 'Goals - home' and 'Goals - away'
# 
# 2) Create another column of "score_diff" to store score difference between Home Team - Away Team (this value can be negative if the Away team is currently winning)

# In[12]:


match_data = match_data[match_data['Goals - home'].notna() & match_data['Goals - away'].notna()].copy()

match_data['score_diff'] = (match_data['Goals - home'] - match_data['Goals - away']).astype(int)


# ### The function filter_out_matches:
# 
# Filters out rows from the DataFrame based on the given conditions. 
# If any row satisfies the rule, all rows with the same fixture_id are excluded.
# 
# Inputs:
# - df: The input DataFrame to be filtered.
# - filters: A dictionary where keys are column names and values are the condition to be applied on that column.
# 
# Returns:
# - A DataFrame with all columns, excluding rows with the matching fixture_id.
# 

# In[13]:


def filter_out_matches(df, filters):
    # Apply each filter in the filters dictionary
    filtered_data = df.copy()  # Create a copy of the original DataFrame to avoid modifying it directly
    for column, condition in filters.items():
        filtered_data = filtered_data.query(f"{column} {condition}")

    # Get the fixture_ids of the rows that satisfy the condition
    filtered_fixture_ids = filtered_data['fixture_id'].unique()

    # Exclude rows with those fixture_ids
    final_filtered_data = df[~df['fixture_id'].isin(filtered_fixture_ids)]

    # Return the filtered data with all columns
    return final_filtered_data.copy()


# ### Selected Rules:
# 
# #### 1) filter1: 
#     if away team gets a Redcard in the first 15 minutes of the game.
#     
# #### 2) filter2: 
#     if home team gets a Redcard in the first 15 minutes of the game.
#     
# #### 3) filter3: 
#     if away team gets a Yellowredcard in the first 15 minutes of the game.
#     
# #### 4) filter4: 
#     if home team gets a Yellowredcard in the first 15 minutes of the game.
#     
# #### 5) filter5: 
#     if away team gets Penalties in the first 15 minutes of the game.
#     
# #### 6) filter6: 
#     if home team gets Penalties in the first 15 minutes of the game.
#     
# #### 7) filter7: 
#     when score_diff is 1 meaning Home team is ahead by 1 score, and Away teams scores a goal in the last 5 minutes of the game.
#     
# #### 8) filter8: 
#     when score_diff is 0 meaning there is a tie, and away teams scores a goal in the last 5 minutes of the game.
#     
# #### 9) filter9: 
#     when score_diff is -1 meaning Away team is ahead by 1 score, and Home teams scores a goal in the last 5 minutes of the game.
#     
# #### 10) filter10: 
#     when score_diff is 0 meaning there is a tie, and Home teams scores a goal in the last 5 minutes of the game.

# In[20]:


filter1 = {
    "halftime": "== '1st-half'",
    "minute": "< 15",
    "`Redcards - away`": "> 0"
}

filter2 = {
    "halftime": "== '1st-half'",
    "minute": "< 15",
    "`Redcards - home`": "> 0"
}

filter3 = {
    "halftime": "== '1st-half'",
    "minute": "< 15",
    "`Yellowred Cards - away`": "> 0"
}

filter4 = {
    "halftime": "== '1st-half'",
    "minute": "< 15",
    "`Yellowred Cards - home`": "> 0"
}

filter5 = {
    "halftime": "== '1st-half'",
    "minute": "< 15",
    "`Penalties - away`": "> 0"
}

filter6 = {
    "halftime": "== '1st-half'",
    "minute": "< 15",
    "`Penalties - home`": "> 0"
}

filter7 = {
    "halftime": "== '2nd-half'",
    "minute": ">= 40",
    "`Score Change - away`": "== 1",
    "score_diff": "== 1"
}

filter8 = {
    "halftime": "== '2nd-half'",
    "minute": ">= 40",
    "`Score Change - away`": "== 1",
    "score_diff": "== 0"
}

filter9 = {
    "halftime": "== '2nd-half'",
    "minute": ">= 40",
    "`Score Change - home`": "== 1",
    "score_diff": "== -1"
}

filter10 = {
    "halftime": "== '2nd-half'",
    "minute": ">= 40",
    "`Score Change - home`": "== 1",
    "score_diff": "== 0"
}


# In[23]:


# List of filters
filters = [filter1, filter2, filter3, filter4, filter5, filter6, filter7, filter8, filter9, filter10]

# Initialize match_data
filtered_matches = match_data
removed_counts = []

# Apply each filter and store the number of matches removed
for f in filters:
    initial_count = len(filtered_matches)
    filtered_matches = filter_out_matches(filtered_matches, f)
    removed_counts.append(initial_count - len(filtered_matches))

removed_counts.append(sum(removed_counts))

# Final filtered matches
normal_matches = filtered_matches


filter_names = [
    "Away Red Card (1st 15 mins)",       
    "Home Red Card (1st 15 mins)",       
    "Away Yellow-Red Card (1st 15 mins)",
    "Home Yellow-Red Card (1st 15 mins)",
    "Away Penalty (1st 15 mins)",        
    "Home Penalty (1st 15 mins)",        
    "Home +1, Away Goal (Last 5 mins)",  
    "Tie, Away Goal (Last 5 mins)",      
    "Away +1, Home Goal (Last 5 mins)",  
    "Tie, Home Goal (Last 5 mins)" ,
    "Total"
]

filter_stat = pd.DataFrame({
    'Filters': filter_names,
    'Number of Removed Matches': removed_counts
})

filter_stat


# Total number of 5636 matches are removed after applying the filters. Especially, for the last 4 filters there is many number of "abnormal" games. Therefore, it might make a difference in the result.

# In[24]:


normal_matches


# In[25]:


# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot for 1st-half normalized=0
plotter1(normal_matches, half='1st-half', normalized=0, ax=axes[0, 0])

# Plot for 1st-half normalized=1
plotter1(normal_matches, half='1st-half', normalized=1, ax=axes[1, 0])

# Plot for 2nd-half normalized=0
plotter1(normal_matches, half='2nd-half', normalized=0, ax=axes[0, 1])

# Plot for 2nd-half normalized=1
plotter1(normal_matches, half='2nd-half', normalized=1, ax=axes[1, 1])

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the grid of plots
plt.show()


# In[26]:


# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot for 1st-half normalized=0
plotter2(normal_matches, half='1st-half', normalized=0, ax=axes[0, 0])

# Plot for 1st-half normalized=1
plotter2(normal_matches, half='1st-half', normalized=1, ax=axes[1, 0])

# Plot for 2nd-half normalized=0
plotter2(normal_matches, half='2nd-half', normalized=0, ax=axes[0, 1])

# Plot for 2nd-half normalized=1
plotter2(normal_matches, half='2nd-half', normalized=1, ax=axes[1, 1])

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the grid of plots
plt.show()


# In[27]:


# Define bins for discretizing P(home win) - P(away win)
bins = np.linspace(-1, 1, 11) 
labels = [f"({bins[i]:.1f}, {bins[i+1]:.1f}]" for i in range(len(bins) - 1)]

# Add a column for the discretized bins
normal_matches['bin'] = pd.cut(
    normal_matches['prob_home'] - normal_matches['prob_away'], 
    bins=bins, 
    labels=labels, 
    include_lowest=True
)

# Calculate the number of games ended as "Draw" and total games in each bin
bin_stats = match_data.groupby('bin').agg(
    draws=('result', lambda x: (x == 'X').sum()),  # Count of "Draw" games
    total_games=('result', 'size'),  # Total games in the bin
    median_prob_draw=('prob_draw', 'median'),  # Median of prob_draw in each bin
    mean_prob_draw=('prob_draw', 'mean')  # Mean of prob_draw in each bin
).reset_index()

# Add a column for the estimated probability of draws
bin_stats['Estimated P(draw)'] = bin_stats['draws'] / bin_stats['total_games']

# Convert the 'P(draw)' column to percentages and round to 2 decimals
bin_stats['Estimated P(draw)'] = (bin_stats['Estimated P(draw)']).round(4)

# Convert 'mean_prob_draw' and 'median_prob_draw' to percentages and round to 4 decimals
bin_stats['mean_prob_draw'] = (bin_stats['mean_prob_draw']).round(4)
bin_stats['median_prob_draw'] = (bin_stats['median_prob_draw']).round(4)

# Display the result as percentages in one line
print(bin_stats.to_string(index=False))


# - There is a very small change in the results of Estimated P(draw) from the previous one when matches are not filtered out. 

# In[29]:


bin_stats_name = pd.DataFrame({
    'Bin': ["(-1.0, -0.8]", "(-0.8, -0.6]", "(-0.6, -0.4]", "(-0.4, -0.2]", 
            "(-0.2, 0.0]", "(0.0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", 
            "(0.6, 0.8]", "(0.8, 1.0]"]})


x = np.arange(len(bin_stats_name['Bin']))
probabilities = np.array(bin_stats['Estimated P(draw)'])
mean_probs = np.array(bin_stats['mean_prob_draw'])

plt.figure(figsize=(10, 6))
plt.bar(x, probabilities, color='skyblue', alpha=0.8, edgecolor='black', label="Estimated P(tie)")

# Adding the line plot
plt.plot(x, mean_probs, color='red', marker='o', linestyle='-', label="Mean Prob Draw")

# Customizing the plot
plt.xticks(x, bin_stats_name['Bin'], rotation=45, ha="right")
plt.xlabel("Bins")
plt.ylabel("Probability")
plt.title("Probability Distribution Across Bins")
plt.legend()
plt.tight_layout()

plt.show()


# Again, the histogram also have a similar behavior as before.

# # Task 3

# First, I removed the features that are non-related to the outcome of the game. Which are:
# - fixture_id
# - latest_bookmaker_update
# - current_time
# - half_start_datetime
# - match_start_datetime
# - suspended
# - stopped
# - name
# - ticking
# - bin

# In[30]:


non_related_features = ['fixture_id','latest_bookmaker_update', 'current_time', 'half_start_datetime', 'match_start_datetime', 'suspended', 'stopped', 'name', 'ticking', 'bin']


# In[31]:


df_tree = normal_matches.drop(columns=non_related_features)


# In[32]:


# remove the actual outcome of the game
df_tree = df_tree.drop(columns=["final_score"])


# I then created a new feature, time_in_minutes, which represents the minute of the game while accounting for half-time information. This feature can be beneficial for the decision tree, as the remaining time in the game significantly impacts the probability of winning or losing.

# In[33]:


df_tree['time_in_minutes'] = df_tree.apply(
    lambda row: row['minute'] + row['second'] / 60 if row['halftime'] == '1st-half' 
    else 45 + row['minute'] + row['second'] / 60, axis=1)

df_tree = df_tree.drop(columns=["minute", "second"])


# In[35]:


# target value should be taken as a categorical variable since it can only take "1", "X", "2" values.
target = "result"
df_tree[target] = df_tree[target].astype('category').cat.codes


# In[36]:


# fixing how some certain features should be treated in the decision tree
df_tree["halftime"] = df_tree["halftime"].apply(lambda half: 1.0 if half == "1st-half" else 2.0)
df_tree["time_in_minutes"] = df_tree["time_in_minutes"].astype(float)
df_tree["current_state"] = df_tree["current_state"].apply(lambda result: 1.0 if result == "1" else 2.0 if result == "2" else 0)


# In[40]:


result = "result"
features = [col for col in df_tree.columns if col != result]

result_df = df_tree[[result]]
feature_df = df_tree[features]


# In[41]:


# creating a Decision Tree with necessary metrics
tree1 = DecisionTreeClassifier(min_impurity_decrease=1e-5, max_depth=5, max_leaf_nodes=10)

tree1.fit(feature_df, result_df)


# In[42]:


# make predictions
predictions = tree1.predict(feature_df)

# calculate performance measures
accuracy = accuracy_score(target_df, predictions)
precision = precision_score(target_df, predictions, average='macro')  
recall = recall_score(target_df, predictions, average='macro')  
f1 = f1_score(target_df, predictions, average='macro')  
conf_matrix = confusion_matrix(target_df, predictions)

# print performance measures
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")


# In[45]:



class_names = target_df.iloc[:, 0].unique().astype(str)

# plot the decision tree
plt.figure(figsize=(20, 15))
plot_tree(tree1, filled=True, feature_names=feature_df.columns, class_names=class_names, rounded=True, fontsize=10)


plt.show()


# The first tree achieves a prediction accuracy of 0.6586, which is considered moderate performance. This can be improved by adding new variables or adjusting the parameters of the decision tree. 
# - increasing depth
# - increasing min_samples_split
# - increasing min_samples_leaf
# may improve the performance of the tree by allowing it to capture more complex patterns in the data. This can lead to better generalization and accuracy. However, we must avoid overfitting, as excessively high values for these parameters can cause the tree to become too complex and sensitive to noise in the data.

# In[46]:


tree2 = DecisionTreeClassifier(
    criterion='gini',  
    max_depth=10,         # increase depth
    min_samples_split=5,  # increase minimum samples to split (default value is 2)
    min_samples_leaf=4,   # increase minimum samples per leaf (default value is 1)
    max_features=None,    
    max_leaf_nodes=20,    
    random_state=42
)
tree2.fit(feature_df, target_df)


# In[47]:


# make predictions
predictions = tree2.predict(feature_df)

# calculate performance measures
accuracy = accuracy_score(target_df, predictions)
precision = precision_score(target_df, predictions, average='macro')  
recall = recall_score(target_df, predictions, average='macro')  
f1 = f1_score(target_df, predictions, average='macro')  
conf_matrix = confusion_matrix(target_df, predictions)

# print performance measures
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")


# In[48]:


class_names = target_df.iloc[:, 0].unique().astype(str)

# plot the decision tree
plt.figure(figsize=(20, 15))
plot_tree(tree2, filled=True, feature_names=feature_df.columns, class_names=class_names, rounded=True, fontsize=10)

plt.show()


# The performance of the tree got slightly better but still not that much high. 

# ### To Compare Feature Importance for Two Decision Trees:

# In[49]:


#feature importance
importances = tree1.feature_importances_

features = feature_df.columns  
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

# sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

importance_df.head(10)


# In[127]:


# feature importance
importances = tree2.feature_importances_


features = feature_df.columns  
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

# sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

importance_df.head(10)


# 
# ### **Feature Importance:**
# 
# - **tree1**:
#   - **Top Features**: 
#     1. `prob_home_norm` (0.121868)
#     2. `prob_away_norm` (0.013666)
#     3. `prob_home` (0.011046)
#     4. `X` (0.006026)
#     5. `Hit Woodwork - home` (0.001791)
# 
#   - The most important feature in tree1 is `prob_home_norm`, which suggests the model heavily relies on the normalized probability of the home team winning. Other important features include the raw probability for the home team (`prob_home`) and the event of hitting the woodwork. Overall, the most important features are related to the odds of home team winning, which can be expected as the odds determined by the fan behavior which most of the time relates to the real probability of winning.
# 
# - **tree2**:
#   - **Top Features**: 
#     1. `prob_home_norm` (0.110764)
#     2. `prob_away` (0.012789)
#     3. `Successful Dribbles - home` (0.010209)
#     4. `Successful Passes - away` (0.006566)
#     5. `Assists - away` (0.001956)
# 
#   - In tree2, `prob_home_norm` is also the most important feature, similar to tree1, though with a slightly lower importance. Additionally, features related to player actions, such as `Successful Dribbles - home` and `Successful Passes - away`, are also significant in determining outcomes.
# 
# ### **Performance Comparison:**
# 
# - **Accuracy**:
#   - **tree1**: 0.6586
#   - **tree2**: 0.6755
#   - Tree2 performs better in terms of accuracy, indicating it classifies more matches correctly.
# 
# - **Precision**:
#   - **tree1**: 0.6369
#   - **tree2**: 0.6625
#   - Tree2 again has a higher precision, meaning it makes fewer false positive predictions, which is important when predicting outcomes like a home win.
# 
# - **Recall**:
#   - **tree1**: 0.6295
#   - **tree2**: 0.6499
#   - Tree2 has better recall, meaning it identifies a higher proportion of actual positive outcomes, which is crucial for capturing all instances of home team wins.
# 
# - **F1 Score**:
#   - **tree1**: 0.6308
#   - **tree2**: 0.6543
#   - Tree2 has a higher F1 score, indicating it balances precision and recall better than tree1.
# 
# 
# ### **Key Differences and Insights**:
# - **Model Complexity**: Tree2 is more complex due to the higher depth (`max_depth=10`) and more splits (`min_samples_split=5`), which might help it capture more intricate relationships in the data. This might explain its better performance across all metrics.
# - **Feature Set**: Both trees prioritize `prob_home_norm` highly, though tree2 places more importance on player actions such as dribbles and passes. This suggests that tree2 benefits from a richer set of features in capturing the game dynamics.
# 
# ### **Conclusion**:
# - Tree2 is the superior model, with better predictive performance and a more nuanced feature importance ranking. The higher depth and additional tuning parameters such as `min_samples_split`, `min_samples_leaf`, and `max_leaf_nodes` allow it to make more accurate and reliable predictions compared to tree1. However, still it can be improved with other additions to achieve a higher accuracy.
# 

# In[ ]:




