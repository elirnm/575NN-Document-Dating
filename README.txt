Everything is set up to be run on patas/condor, where the
corpus used is located.

The files are layed out as follows:
- preprocess.py contains code for preprocessing the data
- split_data.py splits the data into train, dev, and eval portions
- All of the cnn*.py and dense_nn*.py files have implementations
  of various neural network models
- baseline/ contains the implementation of the baselines
- data.py and document_class.py are helper files.
	
Dependencies:
- Python 3
- Keras
- scikit-learn
- NLTK
