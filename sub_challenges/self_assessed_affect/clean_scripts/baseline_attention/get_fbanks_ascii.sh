#!/bin/bash

cwd=${KALDI_ROOT}/egs/compare_2018/self_assessed_intent

source ${cwd}/cmd.sh
source ${cwd}/path.sh

type=$1


# Source the kaldi path
fbank_folder=${cwd}/fbank
target_folder=${cwd}/fbank_ascii_${type}

rm -r ${target_folder}/cleaned/*
rm -r ${target_folder}/raw/*

mkdir -p ${target_folder}/cleaned
mkdir -p ${target_folder}/raw

# Get the mfccs and accomodate in a single file
for file in ${fbank_folder}/raw*.ark
do
 fname=$(basename "$file" .ark)
 cat ${fbank_folder}/${fname}.scp | while read f
 do
   n=`echo "${f}" | cut -d ' ' -f 1`
   echo $f | copy-feats scp:- ark,t:- > ${target_folder}/raw/${n}.fbank # | apply-cmvn scp:${mfcc_folder}/../cmvn_${type}.scp ark:- ark,t:- > ../data/${type}/raw/${n}.mfcc
   cat ${target_folder}/raw/${n}.fbank | sed '/\[$/d' | sed 's/]//g' > ${target_folder}/cleaned/${n}.fbank
 done
done 
