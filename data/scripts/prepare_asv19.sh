#!/usr/bin/env bash

datadir=${1:-"../ASVspoof2019"}
if [[ ! -d $datadir && ! -L $data ]]; then
    #statements
    echo """Input dir ${datadir} does not exist. 
    Please set another dir, where the ASVspoof2019 dataset is located or:
    Run ./download_asv19.sh in this directory"""
    exit
fi
outputdir=${2-"../filelists/asv19"}
mkdir -p $outputdir
echo "Putting filelists to $outputdir"


for subset in LA PA; do
    for protocol in train dev eval;
    do 
        wavedir=$(readlink -e ${datadir}/${subset}/ASVspoof2019_*${protocol}/flac)
        labelfile=$(find -L ${datadir}/${subset}/ASVspoof2019_${subset}_cm_protocols/ -name "*${protocol}*");
        outputfile=${outputdir}/${subset}_${protocol}".tsv"
        cat $labelfile | awk -v base=${wavedir} '
        BEGIN{map["bonafide"]="genuine";print "filename","speaker","systemid","envid","bintype"}
        {print base"/"$2".flac",$1,$3,$4, ($5 in map)?map[$5]:"spoof" }' > $outputfile
        echo "Generated $outputfile"
    done
done
