import csv
import os
import re
import shutil
import sys
from typing import List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EXECUTABLE_DIRECTORY = sys._MEIPASS


# Creates a new directory if it does not already exist
def create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# Moves the files to the target directory and deletes the original files
def move_and_delete_files(source_files: List[str], target_directory: str):
    for file in source_files:
        target_file = os.path.join(target_directory, os.path.basename(file))
        shutil.move(file, target_file)


# Concatenates files and saves the data to a new file
def concat_files(files: List[str], output_file: str) -> None:
    concated_data = pd.concat([pd.read_csv(file, delimiter=',') for file in files])
    if os.path.exists(output_file):
        os.remove(output_file)
    concated_data.to_csv(output_file, index=False)


# Organizes the data files into a structure based on each rank and map
def organize_data_files(file_directory):
    target_directory = os.path.join(file_directory, "CompetitiveData")
    create_directory(target_directory)

    competitive_ranks = ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ascendant", "Immortal", "Radiant"]
    competitive_maps = ["Ascent", "Bind", "Haven", "Split", "Fracture", "Pearl", "Lotus"]

    for rank in competitive_ranks:
        rank_category_directory = os.path.join(target_directory, rank)
        create_directory(rank_category_directory)

        for map in competitive_maps:
            map_directory = os.path.join(rank_category_directory, map)
            create_directory(map_directory)

            pattern = re.compile(f'{rank.lower()}[0-9]*?_{map.lower()}.csv')
            source_files = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if
                            pattern.match(file)]
            target_files = [os.path.join(map_directory, os.path.basename(file)) for file in source_files]

            move_and_delete_files(source_files, map_directory)

            combined_data_file = os.path.join(map_directory, f"{rank}_{map}_CombinedData.csv")
            concat_files(target_files, combined_data_file)


# Constructs the URL for scraping blitz.gg data for each rank and map
def construct_url(rank_number: str, map_name: str) -> str:
    return f"https://blitz.gg/valorant/stats/agents?sortBy=winRate&type=general&sortDirection=DESC&mode" \
           f"=competitive&rank={rank_number}&map={map_name.lower()}"


# Parses the HTML from blitz.gg
def parse_html(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.content, "html.parser")


# Extracts specific data from each row of a blitz.gg URL
def extract_row_data(row) -> List[str]:
    columns = row.find_all("div")
    rank = columns[0].get_text()
    agent = columns[1].get_text()
    kda = columns[3].get_text()
    win_percentage = float(columns[5].get_text().replace("%", "")) / 100
    pick_percentage = float(columns[6].get_text().replace("%", "")) / 100
    average_score = columns[7].get_text()
    first_blood_percentage = float(columns[8].get_text().replace("%", "")) / 100
    matches = columns[9].get_text().replace(",", "").strip('"')

    # Extract kills, deaths, and assists from the KDA column
    kills, deaths, assists = map(float, re.findall(r"\d+\.\d+|\d+", kda))

    return [rank, agent, kills, deaths, assists, win_percentage, pick_percentage, average_score,
            first_blood_percentage, matches]


# Writes the scraped data to a CSV file
def write_to_csv(file_path: str, header: List[str], rows: List[List[str]]):
    with open(os.path.join(EXECUTABLE_DIRECTORY, "..", "..", file_path), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# Scrapes data from blitz.gg for each rank and map
def scrape_data(ranks: List[Tuple[str, str]], maps: List[str]):
    for rank_number, rank_name in ranks:
        for map in maps:
            url = construct_url(rank_number, map)
            soup = parse_html(url)

            selector = '#main-content > div > div.⚡de27659b.inner-wrapper-col > div > div:nth-child(4) > section > ' \
                       'div > div.⚡e728021b.⚡197afe09 > div > div.⚡516a5f38 > div > div'
            rows = soup.select(selector)

            csv_file = f"{rank_name.lower().replace(' ', '')}_{map.lower()}.csv"
            header = ["Rank", "Agent", "Kills", "Deaths", "Assists", "Win %", "Pick %", "Avg. Score", "First Blood %",
                      "Matches"]

            data_rows = [extract_row_data(row) for row in rows]
            write_to_csv(csv_file, header, data_rows)


# Loads the specified rank and map data from the corresponding CSV
def load_data(rank, map_name, file_directory):
    data_directory = os.path.join(file_directory, "CompetitiveData")
    combined_data_file = os.path.join(data_directory, rank, map_name, f"{rank}_{map_name}_CombinedData.csv")
    data = pd.read_csv(combined_data_file)
    return data


# Loads the specified rank and map data from the corresponding CSV and filters the data based on the selected agents
def load_and_filter_data(rank, map_name, team1_agents, team2_agents):
    data_directory = os.path.join(EXECUTABLE_DIRECTORY, "..", "..", "CompetitiveData")
    combined_data_file = os.path.join(data_directory, rank, map_name, f"{rank}_{map_name}_CombinedData.csv")

    data = pd.read_csv(combined_data_file)

    selected_agents = team1_agents + team2_agents
    filtered_data = data[data['Agent'].isin(selected_agents)]

    return filtered_data


# Filters data by selecting specific features and performs one hot encoding
def preprocess_data(filtered_data):
    features = filtered_data[
        ['Agent', 'Kills', 'Deaths', 'Assists', 'Win %', 'Pick %', 'Avg. Score', 'First Blood %', 'Matches']]

    features.loc[:, 'Matches'] = features['Matches'].astype(str).str.extract(r'(\d+)', expand=False).astype(int)

    processed_data = pd.get_dummies(features, columns=['Agent'])

    return processed_data


# Splits data into training and test sets and scales the data
def train_test_split_and_scaling(processed_data):
    X = processed_data.drop('Win %', axis=1)
    y = processed_data['Win %']

    threshold = 0.5
    y_binary = (y >= threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.6, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
    X_test_imputed_scaled = scaler.transform(X_test_imputed)

    return X_train_imputed_scaled, y_train, X_test_imputed_scaled, y_test, X.columns


# Fits regression model with the scaled data and makes prediction
def fit_model(X_train_imputed_scaled, y_train, X_test_imputed_scaled, y_test):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_imputed_scaled, y_train)

    y_pred = model.predict(X_test_imputed_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


# Predicts the winning team and returns the result as a string
def predict_winner(model, team1_agents, team2_agents, feature_names):
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)

    for agent in team1_agents:
        if agent not in team2_agents:
            input_data[f"Agent_{agent}"] = 1
    for agent in team2_agents:
        if agent not in team1_agents:
            input_data[f"Agent_{agent}"] = -1

    prediction = model.predict(input_data.values)

    winning_team = ''
    if prediction == 1:
        winning_team = "Team 1 is predicted to win!"
    elif prediction == 0:
        winning_team = "Team 2 is predicted to win!"

    return winning_team, input_data


# Calculates the average pick rate of the specified agent on the rank/map
def get_pick_rate(agent_name, rank_category, map_name, team_agents):
    data = load_data(rank_category, map_name, os.path.join(EXECUTABLE_DIRECTORY, "..", "..", ))
    agent_data = data[(data['Agent'] == agent_name) & (data['Agent'].isin(team_agents))]
    return agent_data['Pick %'].mean()


# Calculates the average win rate of the specified agent on the rank/map
def get_win_rate(agent_name, rank_category, map_name, team_agents):
    data = load_data(rank_category, map_name, os.path.join(EXECUTABLE_DIRECTORY, "..", "..", ))
    agent_data = data[(data['Agent'] == agent_name) & (data['Agent'].isin(team_agents))]
    return agent_data['Win %'].mean()


# Predicts the winning team based on the rank, map, and agents selected on each team
def get_prediction(rank, map_name, team1_agents, team2_agents):
    filtered_data = load_and_filter_data(rank, map_name, team1_agents, team2_agents)
    features_encoded = preprocess_data(filtered_data)
    X_train_imputed_scaled, y_train, X_test_imputed_scaled, y_test, feature_names = train_test_split_and_scaling(
        features_encoded)
    model, accuracy = fit_model(X_train_imputed_scaled, y_train, X_test_imputed_scaled, y_test)
    winning_team, input_data = predict_winner(model, team1_agents, team2_agents, feature_names)
    probs = model.predict_proba(input_data.values)

    team1_prob = probs[0][1] * 100
    team2_prob = probs[0][0] * 100

    result_string = f"{winning_team}\nPrediction Accuracy: {accuracy * 100:.2f}%"

    return result_string, team1_prob, team2_prob
