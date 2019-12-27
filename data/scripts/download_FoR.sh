#!/usr/bin/env bash

links=("http://www.eecs.yorku.ca/~bil/Datasets/for-norm.tar.gz" "https://www.eecs.yorku.ca/~bil/Datasets/for-orig.tar.gz")

basedir=${1:-../FoR}
echo "Data will be put to ${basedir}" && mkdir -p $basedir

if [[ ! -d $basedir/wav ]]; then
    #statements
    download_tool="$(command -v wget) -c --quiet --show-progress -O"

    echo "Using ${download_tool} to download"
    for data in ${links[@]}; do
        outputpath=${basedir}/$(echo ${data} | awk -F[/?] '{print $(NF)}')
        echo "Downloading ${data} to ${outputpath}"
        $download_tool ${outputpath} ${data}
        echo "Extracting data..."
        tar xzf ${outputpath} -C ${basedir}
    done
fi
