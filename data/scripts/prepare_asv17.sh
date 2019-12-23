#!/usr/bin/env bash

datadir=${1:-"../ASVspoof2017"}
if [[ ! -d $datadir && ! -L $data ]]; then
    #statements
    echo """Input dir ${datadir} does not exist. 
    Please set another dir, where the ASVspoof2017 dataset is located or:
    Run ./download_asv17.sh in this directory"""
    exit
fi
outputdir=${2-"../filelists/asv17"}
mkdir -p $outputdir
echo "Putting filelists to $outputdir"

for protocol in train dev eval;
do 
    wavedir=$(readlink -f ${datadir}/ASVspoof2017_${protocol})
    labelfile=$(find ${datadir}/protocol -name "*${protocol}*");
    cat $labelfile | awk -v base=${wavedir} 'BEGIN{print "filename","speaker","phrase","env_id","play_id","rec_id","bintype"}{print base"/"$1"/"$2, $3,$4,$5,$6,$7,$2 }' > $outputdir/${protocol}".tsv"
    echo "Generated $outputdir/${protocol}.tsv"
done
