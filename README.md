CS:GO Predictive Model  
By JeanKevin56 and KurumiiiiDaKat  

Made in Python with scikit-learn, xgboost and tkinter.  
Datasets from tedtay : 'https://github.com/tedtay/CS-GO-Pro-Matches-Comprehensive-Dataset'  
The datasets are based on Counter-Strike: Global Offensive (CS:GO) and not Counter-Strike 2 (CS2), thus this model contains inaccuracies.  

NOTE : the original datasets have been deleted from the project due to their size. If you want to re-implement them, they can be downloaded at 'https://github.com/tedtay/CS-GO-Pro-Matches-Comprehensive-Dataset/tree/main/scraped_data'. Only 'game_data_rh.csv' and 'historic_games_list.csv' were used. Put both .csv files in the /database/ folder of the project.  
Moreover, the game_data_dataset() function from /database/process_datasets.py must be ran at least once before using the datasets - it cleans up 'game_data_rh.csv' and saves it under a new 'game_data_processed.csv' file.  
Finally, training the model can be done by calling the setup_ml_model() function from the ml_setup.py file.  
