#!/bin/bash

# Run preprocessing and vectorization
python3 PreprocessAndVectorize.py

# Train DNN model
python3 DNN.py

# Train CNN model
python3 CNN.py

# Train LSTM model
python3 LSTM.py

# Test models(Enter your model and path)
python3 EvalTestCustom.py LSTM.pt LSTM 

# Test on custom data (Enter your data path, model path and model)
python3 EvalTestCustom.py data.csv LSTM.pt LSTM

# End of script
