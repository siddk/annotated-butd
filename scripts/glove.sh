# Setup
mkdir data

# Get GloVe Embeddings
wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/GloVe
rm data/glove.6B.zip