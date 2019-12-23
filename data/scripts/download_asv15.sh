#!/usr/bin/env bash

links=("https://datashare.is.ed.ac.uk/bitstream/handle/10283/853/protocol.tar.gz?isAllowed=y" "https://datashare.is.ed.ac.uk/bitstream/handle/10283/853/wav_data.aa.tar.gz?isAllowed=y" "https://datashare.is.ed.ac.uk/bitstream/handle/10283/853/wav_data.ab.tar.gz?isAllowed=y" "https://datashare.is.ed.ac.uk/bitstream/handle/10283/853/wav_data.ac.tar.gz?isAllowed=y")

basedir=${1:-../ASVspoof2015}
mkdir -p $basedir


if [[ ! -d $basedir/wav ]]; then
    #statements
    download_tool="$(command -v wget) -c --quiet --show-progress -O"

    echo "Using ${download_tool} to download"
    for data in ${links[@]}; do
        outputpath=${basedir}/$(echo ${data} | awk -F[/?] '{print $(NF-1)}')
        echo "Downloading ${data} to ${outputpath}"
        $download_tool ${outputpath} ${data}
    done
    # Extracting
    echo "Extracting data"
    tar xzf ${basedir}/protocol.tar.gz -C $basedir
    cat ${basedir}/wav_data.*.tar.gz | tar xzf - -C $basedir
fi
