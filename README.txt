Terms of Service Classification

BERT-based classification model for classifying Terms of Service according to User Privacy grades.

1. Use the parse_json.py script to parse the json files that can be found at https://github.com/tosdr/tosdr.org/tree/master/api/1/service to generate the input training data file, tos_data.csv. The file tos_data.csv has already been created using this script and the json files.
2. Train the model on the data tos_data.csv using the script train.py by running "python train.py"
3. Generate the sentences csv file corresponding to the input file input.txt by running "python preprocess_input_file.py input.txt". This will generate the file test_tos.csv.
4. Use the trained model saved in saved_weights.pt to make predictions on input.txt using the script test.py by running "python test.py"
5. The output will be stored in final_output.json
