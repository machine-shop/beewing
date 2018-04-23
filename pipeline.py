import sys
sys.path.insert(0, './preprocessing')
sys.path.insert(0, './feature_extraction')
sys.path.insert(0, './classification')
import preprocess_image
import random_forest_classifier
import feature_extract
import argparse
"""
Preprocess the bee wing raw images. 
Extract the features from preprocessed bee wing image. 
Train a Random Forest Classifier using extracted features.
Usage:
    python pipeline [-p] [-e] [-t] [-r raw_image_path] [-o output_csv_name] [-tr train_ratio] [-lr learning_rate] [-h hyper hyperparameter]
Options:
    -p preprocess : run preprocess process
    -e feature extraction: run feature extraction process
    -t model training: run model training process
    -r raw_image_path : the input path for raw image
    -o output_csv_name : the name for output features information csv file
    -tr train_ratio : the training ratio for machine learning algorithm
    -lr learning rate : the learning rate for machine learning algorithm
    -h hyper : hyperparameter for algorithm
"""
def main():
     # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Script preprocess the bee wing raw images, extracts the features from preprocessed bee wing image, and trains a Random Forest Classifier using extracted features')
    # Add arguments
    parser.add_argument('-p', '--preprocess', 
        action='store_true',
        help='Run preprocess process')
    parser.add_argument('-e', '--extraction',
        action='store_true',
        help='Run feature extraction process')
    parser.add_argument('-t', '--model_training', 
        action='store_true',
        help='Run model training process')
    parser.add_argument('-r', '--raw_image', 
        type=str, 
        help='The input path for raw image', 
        required=False, 
        default='../raw_image')
    parser.add_argument('-o', '--csv_file', 
        type=str, 
        help='The name for output features information csv file', 
        required=False, 
        default='bee_info.csv')
    parser.add_argument('-tr', '--train_ratio', 
        type=float,
        help='The ratio of dataset used for training', 
        required=False,
        default=0.8)

    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    raw_image_path = args.raw_image
    features_datafile = args.csv_file
    train_ratio = args.train_ratio
    if args.preprocess:
        pipeline_process += ['preprocess']
    if args.extraction:
        pipeline_process += ['extraction']
    if args.model_training:
        pipeline_process += ['model_training']

    if len(pipeline_process) == 0:
        pipeline_process = ['preprocess', 'extraction', 'model_training']
    for step in pipeline_process:
        if step == 'preprocess':  # Proprocess Imgae
            preprocess_raw_image(path)
        elif step == 'extraction':  # Feature extraction
            feature_extract(features_datafile)
        else:  # For Classifcation
            random_forest_classifier(features_datafile, train_ratio)

if __name__ == "__main__":
    main()