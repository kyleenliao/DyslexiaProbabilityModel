import os
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from scipy.stats import beta, norm, bootstrap


def compute_features(Lx, Ly, Rx, Ry, threshold_fixation=0.3, threshold_saccade=1.0):    
    time = np.arange(1, len(Lx) + 1)
                    
    # Compute binocular disparity
    disparity_x = np.abs(Lx - Rx)
    disparity_y = np.abs(Ly - Ry)
    temp = disparity_x**2 + disparity_y**2
    temp = np.asarray(temp, dtype=float)
    avg_disparity = np.mean(np.sqrt(temp))

    # Compute velocity and acceleration
    temp = np.diff(Lx)**2 + np.diff(Ly)**2
    velocity = np.sqrt(np.asarray(temp, dtype=float))
    acceleration = np.diff(velocity)
    median_accel = np.median(acceleration)

    # Identify fixations
    fixation_mask = velocity < threshold_fixation
    fixation_count = np.sum(fixation_mask)
    fixation_duration = np.mean(np.diff(time)[fixation_mask])

    # Identify saccades
    saccade_mask = velocity > threshold_saccade
    saccade_lengths = velocity[saccade_mask]
    avg_saccade_length = np.mean(saccade_lengths) if len(saccade_lengths) > 0 else 0

    # Compute regression rate
    regression_count = np.sum(np.diff(Lx) < 0)
    regression_rate = regression_count / len(Lx)

    return median_accel, avg_disparity, fixation_count, fixation_duration, avg_saccade_length, regression_rate

def clean(arr):
    for i in range(arr.shape[0]):
        arr[i] = float(arr[i].replace(",", "."))
    return arr

def naiveBayes(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Naïve Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Predict probabilities and classes
    y_pred = nb_model.predict(X_test)
    y_prob = nb_model.predict_proba(X_test)[:, 1]  # Probability of being dyslexic

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

def bayesian_probabilities(X, y):
    """Compute posterior probabilities using Bayes' Theorem explicitly."""
    prior_prob = y.mean()  # P(dyslexia)
    likelihoods = {}
    
    for feature in X.columns:
        dyslexic_group = X[y == 1][feature]
        control_group = X[y == 0][feature]
        
        # Fit normal distributions to each group
        mu_d, sigma_d = dyslexic_group.mean(), dyslexic_group.std()
        mu_c, sigma_c = control_group.mean(), control_group.std()
        
        # Avoid division by zero
        if sigma_d == 0: sigma_d = 1e-6
        if sigma_c == 0: sigma_c = 1e-6
        
        likelihoods[feature] = (mu_d, sigma_d, mu_c, sigma_c)
    
    def compute_posterior(sample):
        """Calculate posterior probability for a given sample."""
        posteriors = []
        for feature in X.columns:
            mu_d, sigma_d, mu_c, sigma_c = likelihoods[feature]
            p_x_given_d = norm.pdf(sample[feature], mu_d, sigma_d)
            p_x_given_c = norm.pdf(sample[feature], mu_c, sigma_c)
            posterior = (p_x_given_d * prior_prob) / (p_x_given_d * prior_prob + p_x_given_c * (1 - prior_prob))
            posteriors.append(posterior)
        return np.mean(posteriors)  # Average probability across features

    return X.apply(compute_posterior, axis=1)

def main():
    # Define the root folder where all subject folders are located
    root_folder = "/Users/kyleenliao/Downloads/RecordingData"  # Change this to your actual folder path

    # List to store all data
    all_data = []

    # Iterate over each folder in the root directory
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        # Ensure it's a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Extract subject ID and classify based on last digit
        subject_code = folder_name[-1]
        
        if subject_code in "12":
            reading_status = "reading_disabled"
        elif subject_code in "34":
            reading_status = "control"
        else:
            continue  # Skip folders with unexpected names

        gender = "male" if subject_code in "13" else "female"

        # Find the text file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):  # Assuming the file is a .txt file
                file_path = os.path.join(folder_path, file_name)
                
                # Read the file into a DataFrame
                df = pd.read_csv(file_path, delimiter="\t")
                
                LX_array = clean(np.asarray(df["LX"]))
                LY_array = clean(np.asarray(df["LY"]))
                RX_array = clean(np.asarray(df["RX"]))
                RY_array = clean(np.asarray(df["RY"]))
                                
                median_accel, avg_disparity, fixation_count, fixation_duration, avg_saccade_length, regression_rate = compute_features(LX_array, LY_array, RX_array, RY_array)
                
                # Add subject metadata to the DataFrame
                subject_data = {
                    #"Subject_ID": folder_name,
                    "Reading_Status": reading_status,
                    #"Gender": gender,
                    'Median_Acceleration': median_accel,
                    'Avg_Disparity': avg_disparity,
                    'Fixation_Count': fixation_count,
                    #'Fixation_Duration': fixation_duration,
                    'Avg_Saccade_Length': avg_saccade_length,
                    'Regression_Rate': regression_rate
                }
                
                # Append data
                all_data.append(subject_data)

    # Combine all data into a single DataFrame
    final_df = pd.DataFrame(all_data)
    final_df['Reading_Status_Binary'] = final_df['Reading_Status'].map({"control": 0, "reading_disabled": 1})
    
    features = ['Median_Acceleration', 'Avg_Disparity', 'Fixation_Count', 
            #'Fixation_Duration', 
            'Avg_Saccade_Length', 'Regression_Rate']

    correlations = {feat: pointbiserialr(final_df[feat], final_df['Reading_Status_Binary'])[0] for feat in features}
    correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
    print(correlation_df.sort_values(by='Correlation', ascending=False))
    
    final_df = final_df.drop("Reading_Status", axis=1)
        
    y = final_df["Reading_Status_Binary"]
    X = final_df.drop("Reading_Status_Binary", axis=1)
    
    naiveBayes(X,y)
    
    posterior_probs = bayesian_probabilities(X, y)
    print("Posterior Probabilities:")
    print(posterior_probs)
    
    brier = brier_score_loss(y, posterior_probs)
    print(f"Brier Score: {brier:.4f}")
    
    threshold = 0.5  
    predictions = (posterior_probs > threshold).astype(int)
    # Compute accuracy
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()