Overview

This project is designed for advanced analysis and classification of peptide sequences, focusing on predicting bioactivity. The code extracts features, applies multi-channel data representation, and constructs a machine learning model for classification, specifically using the Deep Forest (Cascade Forest Classifier) algorithm. Key functionalities include amino acid composition analysis, physicochemical feature extraction, and conversion of features into RGB image-like data for enhanced representation.

Features

Feature Extraction from Peptides:
Calculates amino acid properties, n-grams, group-based features, and PAAC (Pseudo Amino Acid Composition) using methods like cksaagp and calculate_PAAC.
Computes physicochemical properties (e.g., molecular weight, isoelectric point).
RGB Image-like Feature Construction:
Converts extracted features into R, G, and B channels.
Reshapes and combines these channels into 2D matrices representing RGB images, enhancing the model’s ability to capture spatial patterns.
Deep Forest Model:
The Cascade Forest Classifier, a deep forest model, is trained using Stratified K-Fold cross-validation.
Performance metrics include accuracy, sensitivity, precision, F1 score, MCC, and AUC.
Dependencies

This code requires the following Python libraries:

BioPython: for sequence parsing and manipulation
modlamp: for peptide descriptor calculations
scikit-learn: for model selection and metrics
deepforest: for the Cascade Forest Classifier
pandas and numpy: for data handling and numerical operations
To install, run the command: pip install biopython modlamp scikit-learn deepforest pandas numpy

Usage

Data Preparation:
Input peptide sequences in a FASTA format or CSV files containing sequence information.
Run the read_fasta function to parse sequences into a pandas DataFrame.
Use construct_features to generate comprehensive feature sets from sequence data.
Model Training:
Use main() to initiate training and evaluation.
extract_ggap_features and create_rgb_image_features are used to prepare and flatten RGB-like data from peptide features.
The Cascade Forest Classifier is trained with cross-validation for robust performance evaluation.
Evaluation:
Outputs include a confusion matrix, accuracy, sensitivity, precision, F1 score, MCC, and AUC.
Use model predictions on test data to evaluate model generalizability.
Example Commands

To execute the full pipeline, run: python main.py

This script will read input data, extract features, generate RGB-like features, train the model with cross-validation, and output evaluation metrics.

Note

Future model improvements can include diversifying training datasets, fine-tuning model hyperparameters, and experimenting with additional feature extraction methods to optimize accuracy and adaptability.
