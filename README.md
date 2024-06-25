# Multi-Author Writing Style Analysis: a tentative approach between Transformers and traditional NLP

### This Repo can work with the correct weight matrices for all the trained models and with the dataset as shared by PAN in 2024. I am not pushing the weight matrices, but you can retrain the models following the instructions. You will need to dump stylometry_extractor.pkl through the features.ipynb notebook, so it can be used by the Siamese network!

### Here are the instructions to replicate my experiments and informations on my Repo structure. It can get a bit complex, if you need help feel free to contact me at lqv142@ku.alumni.dk! 

## Repo structure:

* data -> Contains raw dataset from PAN and already parsed data in csv form 
* weights -> Contains all the weights of our trained model. DeBERTa has one weight for each of the 3 trained models, Siamese network directory contains weight for Sentence BERT and gensim's topic models
* DeBertaV3.ipynb -> Notebook for DeBERTa training
* Siamese Full Network.ipynb -> Notebook for Siamese Network training
* Model_complete.ipynb -> Notebook for running evaluation, collecting predictions from both models and their combination, running error analysis. It is our "main" notebook. 
* features.ipynb and features.py -> Notebook and py file for experimenting and passing the stylometry extractor to Siamese network
* redditparser_ntbk.ipynb and redditparser.py -> Notebook and py file for parsing PAN data, experimenting and passing the csvs to be further processed by our models
* stylometry_extractor.pkl -> Pickle file containing our stylometry extractor, was useful because I had to work both on google Colab and local. The extractor gets passed to our model through this file (but you can change with a simple import from features.py since now the code is all in one place)
* plt.ipynb -> The notebook for the plot of our results. I take notes for the results by hand and then hardcoded the floating points in the notebook, because I had to run different inferences across different difficulties, and getting everything on the same csv/json was a bit troublesome. In any case you can replicate the results running Model_complete.ipynb and getting comparable results.  
* sbert.py and sentiment.py -> Little helper functions, legacy 
* corpus.txt -> txt with all the paragraphs from the dataset in a single place. Was mainly used during feature extraction in features.ipynb

## Instructions to replicate experiments:

* Make sure that all libraries are installed in the environment and that all datasets are correctly in place in the repository. Pay attention to this step each time a new notebook is used. 
* _[Optional: re-parse the data]_
    * _Open redditparser_ntbk.ipynb_
    * _Run notebook. The dataset is parsed in 6 different tables, three couples per difficulty. Each difficulty has a training csv and a test/eval csv._
    * _Some of the cells can be a bit confusing: that's because the notebook is "legacy" and I didn't have to clean it up after the first successful parsing; then I ran experiments on some notebook cells and things got a bit convoluted here and there._
* _[Optional: re-train the models]_
    * _Re-train the Siamese network._ 
        * _Open Siamese_Full_Network.ipynb and set up the difficulty. Change all the appropriate variables according to the difficulty (TRAIN_CSV, TEST_CSV PATH and lda_model); change the threshold value according to the difficulty (0.55 hard, 0.6 medium, 0.7 easy); change the contrastive loss margin according to the difficulty (0.0 hard, 0.15 medium, 0.10 easy); change the dropout rate according to the difficulty (0.2 hard, 0.4 medium, 0.5 easy)._
        * _Uncomment the last three cells to save the weights._
        * _Run all cells from the notebook_ 
    * _Re-train DeBERTa (I mean, fine-tune)._
        * _Open DeBertaV3.ipynb and set up the difficulty. Change all the appropriate variables according to the difficulty (paths)._
        * _Access through own wandb key or just ignore/comment lines concerning wandb_
        * _Run all cells the notebook_
* You will find the data from the shared task saved in csv form in the 'data' directory. The 'eval' tables are the ones that will be passed to the models to process the paragraphs.
* You will find the pre-trained weights in the 'weights' directory. The weights are divided per model and difficulty. The combined model uses a late decision rule so it's strictly dependent on the individual models' weight, without additional training.
* Open Model_complete.ipynb and set up the difficulty. Change all the appropriate variables according to the difficulty (TEST_CSV, PATH for the Siamese Network, paths for DeBERTa, and lda_model). Change the threshold value for the Siamese network (0.55 hard, 0.6 medium, 0.7 easy).
* Run all cells from Model_complete.ipynb. This will:
    1. Preprocess the text for the Siamese network
    2. Get the inference for the Siamese network with the accuracy reports
    3. Preprocess the text for DeBERTa 
    4. Get the inference for DeBERTa with the accuracy reports
    5. Combine the probabilities extracted from the two models and get the combined predictions
    6. Run the error analysis described in the paper