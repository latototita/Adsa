import requests
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Fetch data for all available leagues over the past 10 years
url = "https://api.football-data.org/v2/competitions"
headers = {
    "X-Auth-Token": "YOUR_API_KEY"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    competitions = response.json()["competitions"]

    # Filter and retrieve the desired leagues
    desired_leagues = ["PL", "BL1", "FL1", "PD", "SA"]  # Premier League, Bundesliga, Ligue 1, La Liga, Serie A
    league_data = []

    for competition in competitions:
        if competition["code"] in desired_leagues:
            league_id = competition["id"]
            league_name = competition["name"]
            league_data.append({"id": league_id, "name": league_name})
else:
    print("Error:", response.status_code)

# User input for match prediction
home_team = input("Enter the name of the home team: ")
away_team = input("Enter the name of the away team: ")

# Function to fetch historical match data for a specific league
def fetch_match_data(league_id):
    url = f"https://api.football-data.org/v2/competitions/{league_id}/matches"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        match_data = response.json()["matches"]
        return match_data
    else:
        print("Error:", response.status_code)
        return []

# Fetch match data for each league
match_data_all_leagues = []

for league in league_data:
    league_id = league["id"]
    league_name = league["name"]
    print(f"Fetching match data for {league_name}...")
    match_data = fetch_match_data(league_id)
    match_data_all_leagues.extend(match_data)

# Preprocess the data, split into features and labels, etc.
# ...

# Extract features and labels from the match data
labels = []
features = []

for match in match_data_all_leagues:
    # Extract the label (outcome) of the match
    home_goals = match["score"]["fullTime"]["homeTeam"]
    away_goals = match["score"]["fullTime"]["awayTeam"]
    if home_goals > away_goals:
        labels.append(1)  # Home team win
    elif home_goals < away_goals:
        labels.append(2)  # Away team win
    else:
        labels.append(0)  # Draw

    # Extract the features of the match
    # Example: Extracting home team's average goals scored and conceded
    home_team_stats = match["homeTeam"]["teamStats"]["home"]
    home_avg_goals_scored = home_team_stats["goalsScored"]["avg"]
    home_avg_goals_conceded = home_team_stats["goalsConceded"]["avg"]

    # Example: Extracting away team's average goals scored and conceded
    away_team_stats = match["awayTeam"]["teamStats"]["away"]
    away_avg_goals_scored = away_team_stats["goalsScored"]["avg"]
    away_avg_goals_conceded = away_team_stats["goalsConceded"]["avg"]

    # Example: Creating a feature vector for the match
    match_features = [home_avg_goals_scored, home_avg_goals_conceded,
                      away_avg_goals_scored, away_avg_goals_conceded]

    features.append(match_features)

# Convert the features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define and train an ensemble of models
n_models = 10000  # Number of models in the ensemble
ensemble_models = []

for _ in range(n_models):
    # Create and train your individual model (e.g., decision tree, random forest, logistic regression)
    model = YourModel()  # Replace YourModel with the appropriate scikit-learn classifier
    model.fit(X_train, y_train)
    ensemble_models.append(model)

# Make predictions using the ensemble of models
ensemble_predictions = []

for model in ensemble_models:
    predictions = model.predict(X_test)
    ensemble_predictions.append(predictions)

# Combine predictions using simple averaging
ensemble_predictions = np.array(ensemble_predictions)
combined_predictions = np.mean(ensemble_predictions, axis=0)

# Convert the average predictions to integer scores (optional)
combined_predictions = combined_predictions.astype(int)

# Print the predicted outcomes and scores
print("Predicted outcomes and scores:")
print(classification_report(y_test, combined_predictions))

# Predict the outcome and scores for the user-input match
user_match_features = preprocess_user_input(home_team, away_team)
user_match_predictions = []

for model in ensemble_models:
    prediction = model.predict(user_match_features)
    user_match_predictions.append(prediction)

user_match_predictions = np.array(user_match_predictions)
user_combined_prediction = np.mean(user_match_predictions, axis=0)

# Convert the average predictions to integer scores (optional)
user_combined_prediction = user_combined_prediction.astype(int)

# Print the predicted outcome and scores for the user-input match
predicted_home_score = user_combined_prediction[0]
predicted_away_score = user_combined_prediction[1]

print(f"\nPredicted outcome for {home_team} vs. {away_team}:")
if predicted_home_score > predicted_away_score:
    print(f"Winner: {home_team}")
    print(f"Loser: {away_team}")
elif predicted_home_score < predicted_away_score:
    print(f"Winner: {away_team}")
    print(f"Loser: {home_team}")
else:
    print("It's a draw!")

print(f"Predicted scores:")
print(f"{home_team}: {predicted_home_score}")
print(f"{away_team}: {predicted_away_score}")
