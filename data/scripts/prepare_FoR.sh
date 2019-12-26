#!/usr/bin/env bash

datadir=${1:-"../FoR/"}
if [[ ! -d $datadir && ! -L $data ]]; then
    #statements
    echo """Input dir ${datadir} does not exist. 
    Please set another dir, where the FoR dataset is located or:
    Run ./download_FoR.sh in this directory"""
    exit
fi
outputdir=${2-"../filelists/FoR"}
mkdir -p $outputdir
echo "Putting filelists to $outputdir"

function filter_length() {
    # In the dataset there are at least 2 samples being empty
    length=$(soxi -D $1)
    echo $length $1 | awk '$1>0{print $2}'
}

export -f filter_length

pipe_command="parallel --bar"



wavedir=$(readlink -e ${datadir}) # Full path to waves
for data in $(find -L ${datadir} -type d -name for-*); do
    echo "Processing ${data}. Filtering for non-zero files..."
    subset_name=${data##*/};
    data=$(readlink -f $data)
    # Train
    find ${data}/training/fake/ -name *.wav | $pipe_command filter_length | awk 'BEGIN{print "filename","bintype"}{print $1,"spoof"}' > $outputdir/${subset_name}"_train.tsv"
    find ${data}/training/real/ -name *.wav | $pipe_command filter_length | awk '{print $1,"genuine"}' >> $outputdir/${subset_name}"_train.tsv"
    # DEV
    find ${data}/validation/fake/ -name *.wav |$pipe_command filter_length | awk 'BEGIN{print "filename","bintype"}{print $1,"spoof"}' > $outputdir/${subset_name}"_dev.tsv"
    find ${data}/validation/real/ -name *.wav | $pipe_command filter_length|awk '{print $1,"genuine"}' >> $outputdir/${subset_name}"_dev.tsv"
    # Eval
    find ${data}/testing/fake/ -name *.wav | $pipe_command filter_length | awk 'BEGIN{print "filename","bintype"}{print $1,"spoof"}' > $outputdir/${subset_name}"_eval.tsv"
    find ${data}/testing/real/ -name *.wav | $pipe_command filter_length | awk '{print $1,"genuine"}' >> $outputdir/${subset_name}"_eval.tsv"
    echo "Generated labels to $outputdir/${subset_name}*"
done
