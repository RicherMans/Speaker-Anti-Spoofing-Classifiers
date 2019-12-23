#!/usr/bin/env bash

links=("https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?isAllowed=y" "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/PA.zip?isAllowed=y")

basedir=${1:-../ASVspoof2019}
mkdir -p $basedir


if [[ ! -d $basedir/wav ]]; then
    #statements
    download_tool="$(command -v wget) -c --quiet --show-progress -O"

    echo "Using ${download_tool} to download"
    for data in ${links[@]}; do
        outputpath=${basedir}/$(echo ${data} | awk -F[/?] '{print $(NF-1)}')
        echo "Downloading ${data} to ${outputpath}"
        $download_tool ${outputpath} ${data}
        echo "Extracting ${outputpath}..."
        unzip -qq ${outputpath} -d $basedir
    done
fi
