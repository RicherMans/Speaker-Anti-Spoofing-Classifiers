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
    wavedir=$(readlink -e ${datadir}/ASVspoof2017_*${protocol})
    if [[ ${protocol} == "eval" ]]; then
        # Evaluation has 2 protocols
        labelfile=$(find -L ${datadir}/protocol -name "*${protocol}*"); # Two files will be found
        paste $labelfile | awk -v base=${wavedir} '
        BEGIN{print "filename","speaker","phrase","env_id","play_id","rec_id","bintype"}
        {print base"/"$1, $5,$4,$7,$8,$9,$2 }' > $outputdir/${protocol}".tsv"
    else
        labelfile=$(find -L ${datadir}/protocol -name "*${protocol}*");
        cat $labelfile | awk -v base=${wavedir} '
        BEGIN{print "filename","speaker","phrase","env_id","play_id","rec_id","bintype"}
        {print base"/"$1,$3,$4,$5,$6,$7,$2 }' > $outputdir/${protocol}".tsv"
    fi
    echo "Generated $outputdir/${protocol}.tsv"
done
