from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import random

app = Flask(__name__)

# Generate random sample data connected with 'Asur' web series
characters = ['Dhananjay Rajput', 'Nikhil Nair', 'Lolark Dubey', 'Rasool', 'Rhea']
times_of_day = ['Night', 'Evening', 'Morning', 'Afternoon']
locations = ['Lab', 'Office', 'Home']
previous_actions = ['Research', 'Meeting', 'Sleeping', 'Observing', 'Reading', 'Experimenting', 'Testing']
next_actions = ['Analyzing', 'Discussing', 'Waking up', 'Recording', 'Writing', 'Testing']

data = {
    'Character': [random.choice(characters) for _ in range(20)],
    'Time_of_Day': [random.choice(times_of_day) for _ in range(20)],
    'Location': [random.choice(locations) for _ in range(20)],
    'Previous_Action': [random.choice(previous_actions) for _ in range(20)],
    'Next_Action': [random.choice(next_actions) for _ in range(20)]
}

df_asur = pd.DataFrame(data)

# Initialize LabelEncoders
le_character = LabelEncoder()
le_time = LabelEncoder()
le_location = LabelEncoder()
le_prev_action = LabelEncoder()
le_next_action = LabelEncoder()

# Apply LabelEncoders to categorical columns
df_asur['Character'] = le_character.fit_transform(df_asur['Character'])
df_asur['Time_of_Day'] = le_time.fit_transform(df_asur['Time_of_Day'])
df_asur['Location'] = le_location.fit_transform(df_asur['Location'])
df_asur['Previous_Action'] = le_prev_action.fit_transform(df_asur['Previous_Action'])
df_asur['Next_Action'] = le_next_action.fit_transform(df_asur['Next_Action'])

# Separate features and target variable
X = df_asur[['Character', 'Time_of_Day', 'Location', 'Previous_Action']]
y = df_asur['Next_Action']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_next_action():
    character = request.form['character']
    time_of_day = request.form['time_of_day']
    location = request.form['location']
    prev_action = request.form['previous_action']
    
    # Encode the input using the same encoders used for training
    character_encoded = le_character.transform([character])[0]
    time_encoded = le_time.transform([time_of_day])[0]
    location_encoded = le_location.transform([location])[0]
    prev_action_encoded = le_prev_action.transform([prev_action])[0]
    
    # Make a prediction using the trained model
    predicted_next_action = clf.predict([[character_encoded, time_encoded, location_encoded, prev_action_encoded]])
    
    # Decode the predicted next action
    predicted_next_action_decoded = le_next_action.inverse_transform(predicted_next_action)
    
    # Render result.html with the prediction
    return render_template('result.html', prediction=predicted_next_action_decoded[0])


if __name__ == '__main__':
    app.run(debug=True)
