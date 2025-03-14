# Dyslexia Classification Using Eye-Tracking Data

Dyslexia is a common learning difference that affects reading fluency, comprehension, and spelling, yet it often goes undiagnosed until later stages of education. Early detection is crucial for providing timely interventions that can significantly improve academic and personal outcomes for individuals with dyslexia. Traditional diagnostic methods often rely on behavioral assessments, which can be time-consuming, subjective, and require trained professionals.

Eye-tracking technology offers an objective and non-intrusive way to analyze reading patterns, providing a rich source of data on how individuals process written text. Research has shown that people with dyslexia exhibit distinct eye movement patterns, such as increased fixations, longer gaze durations, and more frequent regressions (re-reading of words or phrases). By leveraging this data, machine learning and probabilistic models can help automate dyslexia detection, making it faster, more scalable, and more accessible to diverse populations.

The dataset used in this work is from "Screening for Dyslexia Using Eye Tracking during Reading" conducted by Nilsson Benfatto and his team. This dataset has 98 dyslexic candidates and 88 control candidates and included the X and Y coordinates of the left and right eyes. 

This project explores the application of probabilistic models (Naive Bayes and Logistic Regression) to classify dyslexia using eye-tracking data. The primary objective is to extract features from binocular eye movements and compare the performance of various classifiers in distinguishing individuals with dyslexia from controls. The study further assesses model fairness across gender groups.
