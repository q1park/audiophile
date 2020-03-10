#!/bin/bash

FILES=words/*

for f in ${FILES}
do
    file=$(echo "${f}" | grep -oP '(?<=\/)\w+(?=\.)')
    python xml2rttm.py words/${file}.*.words.xml | sort -n -k 4 > rttm/${file}.rttm
    echo "Processing ${file}"
done
