Dataset Name: Corrosion Dataset

There are two CSV files: Corrosion_train.csv and Corrosion_test.csv.
Each CSV file has five columns: filename, S11, S21, Phase11, and Phase21.
filename: A string representing the original source filename without the .txt extension.
S11, S21, Phase11, Phase21: Each of these columns is a single string containing 201 space-separated float values.


All target (ground-truth) images are located under ./datasets/corrosion_img.
The target image for a sample's mesurement record can be referenced using the filename column value in the csv file. The filename is using this convention: [DATE:MMDD]_[SAMPLE_INDEX]_[CORROSION_VALUE]_[real|augmented]. For example, if the 'filename' column value was '0525_61_30.89263840450541_augmented', the diffusion model should try to match the image 'datasets/corrosion_img/61/0525_61_30.89263840450541_augmented.png' using the values in other columns. 