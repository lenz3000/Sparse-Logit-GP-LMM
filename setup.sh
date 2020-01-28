#!/bin/bash
echo
echo 'Downloading data  '
echo
wget -N 'https://tubcloud.tu-berlin.de/s/RLXKE6fHgrtrmay/download'
echo 'Unpacking data .. '
echo
unzip download
rm download
mkdir Images
mkdir Models
mkdir logs
