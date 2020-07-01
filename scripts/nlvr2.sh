# Setup
mkdir data
mkdir data/NLVR2-Features
mkdir data/NLVR2-Questions

# Get NLVR 2 Features
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/train_obj36.zip
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/valid_obj36.zip
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/test_obj36.zip

unzip train_obj36.zip
unzip valid_obj36.zip
unzip test_obj36.zip

rm train_obj36.zip
rm valid_obj36.zip
rm test_obj36.zip

mv nlvr2_imgfeat/train_obj36.tsv data/NLVR2-Features
mv nlvr2_imgfeat/valid_obj36.tsv data/NLVR2-Features
mv nlvr2_imgfeat/test_obj36.tsv data/NLVR2-Features

rm -r nlvr2_imgfeat

# Get NLVR 2 Questions
wget https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/train.json
wget https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/dev.json
wget https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/test1.json

mv *.json data/NLVR2-Questions