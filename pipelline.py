import sys
sys.path.insert(0, './preprocessing')
sys.path.insert(0, './feature_extraction')
sys.path.insert(0, './classification')
import preprocess_image
import random_forest_classifier
import feature_extract

"""
Preprocess the bee wing raw images. 
Extract the features from preprocessed bee wing image. 
Train a Random Forest Classifier using extracted features.

options:
    -p preprocess : run preprocess process
    -e feature extraction: run feature extraction process
    -t model training: run model training process
    -r raw_image_path : the path for input path for raw image
    -o output_csv_name : the name for output features information csv file
    -tr train_ratio : the training ratio for machine learning algorithm
    -lr learning rate : the learning rate for machine learning algorithm
    -h hyper : hyperparameter for algorithm
"""
def main():
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'p:e:t:r:o:tr:', ['preprocess=', 'feature extraction=', 'model training=', 'raw_image_path=', 'output_csv_name=', 'train_ratio='])
    except getopt.GetoptError:
        print("usage")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-r', '--raw_image_path'):
            raw_image_path = arg
        if opt in ('-o', '--output_csv_name'):
            features_datafile = arg
        if opt in ('-tr', '--train_ratio'):
            train_ratio = arg
    #     raw_image_path = "../raw_image"
    #     features_datafile = 'bee_info.csv'
    #     train_ratio = 0.8

    # Proprocess Imgae
    preprocess_raw_image(path)

    # Feature extraction
    feature_extract(features_datafile)

    # For Classifcation
    random_forest_classifier(features_datafile, train_ratio)


if __name__ == "__main__":
    main()