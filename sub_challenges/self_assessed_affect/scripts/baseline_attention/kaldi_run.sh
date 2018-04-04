#!/bin/bash

#### Create Stuff
mkdir -p data/train data/devel data/test
mkdir -p local

#### Copy stuff
ln -s ../../wsj/s5/steps/
ln -s ../../wsj/s5/utils/
cp ../../wsj/s5/cmd.sh .
cp ../../wsj/s5/path.sh .
cp -r ../../wsj/s5/conf/ .
cp ../../voxforge/s5/conf/decode.config conf
cp ../../yesno/s5/local/score.sh local/
cp -r ../../gale_arabic/s5/local/nnet local


#### Source stuff
. ./cmd.sh
. ./path.sh

rm -r data/train/*

# Create wav.scp
rm -f data/train/wav.scp
for file in /home3/srallaba/data/ComParE2018_SelfAssessedAffect/wav/train_*
  do 
     fname=$(basename "$file" .wav) 
     echo sai-$fname $file >> data/train/wav.scp
  done
echo "Generated train wav.scp"

for file in /home3/srallaba/data/ComParE2018_SelfAssessedAffect/wav/devel_*
  do 
     fname=$(basename "$file" .wav) 
     echo sai-$fname $file >> data/devel/wav.scp
  done
echo "Generated devel wav.scp"

for file in /home3/srallaba/data/ComParE2018_SelfAssessedAffect/wav/test_*
  do 
     fname=$(basename "$file" .wav) 
     echo sai-$fname $file >> data/test/wav.scp
  done
echo "Generated test wav.scp"


# Generate utt2spk
cut -d ' ' -f 1 data/train/wav.scp > utterances.train
cut -d ' ' -f 1 data/train/wav.scp  > speakers.train
paste utterances.train speakers.train > data/train/utt2spk

cut -d ' ' -f 1 data/devel/wav.scp > utterances.devel
cut -d ' ' -f 1 data/devel/wav.scp  > speakers.devel
paste utterances.devel speakers.devel > data/devel/utt2spk

cut -d ' ' -f 1 data/test/wav.scp > utterances.test
cut -d ' ' -f 1 data/test/wav.scp  > speakers.test
paste utterances.test speakers.test > data/test/utt2spk

# Generate spk2utt
./utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
./utils/utt2spk_to_spk2utt.pl data/devel/utt2spk > data/devel/spk2utt 
./utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt 
 
./utils/fix_data_dir.sh data/train 
./utils/fix_data_dir.sh data/test
./utils/fix_data_dir.sh data/devel


# Set the right conf file
fbankdir=fbank
# Generate the fbank features; by default 40-dimensional fbanks on each frame
for set in train devel test; do
  steps/make_fbank.sh --cmd "run.pl" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
  utils/fix_data_dir.sh data/$set || exit;
  steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
done


./local/get_fbanks_ascii.sh train
./local/get_fbanks_ascii.sh devel
./local/get_fbanks_ascii.sh test



