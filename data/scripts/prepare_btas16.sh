#!/usr/bin/env bash

datadir=${1:-"../BTAS16"}
if [[ ! -d $datadir && ! -L $data ]]; then
    #statements
    echo """Input dir ${datadir} does not exist. 
    Please set another dir, where the BTAS16 dataset is"""
    exit
fi
outputdir=${2-"../filelists/btas16"}
mkdir -p $outputdir
echo "Putting filelists to $outputdir"

for protocol in train dev;
do 
    wavedir=$(readlink -f ${datadir}/${protocol})
    labelfile=$(find ${datadir} -name "*${protocol}*-list.txt" );
    cat $labelfile | awk -v base=${wavedir} 'BEGIN{map["genuine"]="genuine"; 
    print "filename","speaker","patype","bintype"}
    {print base"/"$2, $1, $3, ($3 in map)?map[$3]:"spoof" }' > $outputdir/${protocol}".tsv"
    echo "Generated $outputdir/${protocol}.tsv"
done

# Treat evaluation data differently ... evaluation data also encompassed some development data, so remove it
wavedir=$(readlink -f ${datadir}/test)
# Test data needs to be derandomized and some anchors removed from the competition
awk -v base=${wavedir} -F[' '.] 'FILENAME==ARGV[1]{map[$1]=$2;}
FILENAME==ARGV[2]{val=map[$1]; DELETE_VALUE[val]}
FILENAME==ARGV[3]{val=map[$1]; DELETE_VALUE[val];} 
FILENAME==ARGV[4]{if(!($2 in DELETE_VALUE)){print}} 
FILENAME==ARGV[5]{if(!($2 in DELETE_VALUE)){print}}' ${datadir}/btas2016-testset-match-names.txt ${datadir}/real-dev-anchors4test.txt ${datadir}/attack-dev-anchors4test.txt ${datadir}/test-real-list-derandomized.txt ${datadir}/test-attack-list-derandomized.txt > $outputdir/eval.tsv

echo "Generated $outputdir/eval.tsv"
