cd category_extraction
python CnnCategoryExtractor.py > output/output_cnn.txt

cd ../ote
python OpinionTargetExtractor.py > output/output_lstm.txt

cd ../sentiment_polarity
python SentimentPolarityClassifier.py > output/output_cnn.txt