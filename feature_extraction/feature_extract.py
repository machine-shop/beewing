def feature_extract(features_datafile):
    
    
    feature_list = []
    # Suppose the data feature is stored in multiple lists / arrays
    file = open(features_datafile, 'w')
    # format dummy header
    file.write('{}, {}, {}, {}, {}'.format('one', 'two', 'three', 'four', 'five')
    file.write("\n") 

    """
    Pseudocode for save the extracted feature into file, the details of code
    will need to be updated according to the feature extraction code

    For each image in the folder
        extract features
        save features into lists
        write to file
    """

    
    feature_line = ','.join(feature_list) + '\n'
    # file_write(feature_list)

    file.close()