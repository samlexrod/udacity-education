#!/bin/bash

# Create folder and download dog dataset
mkdir dogImages
cd dogImages
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip
rm -f dogImages.zip

echo "Dog dataset downloaded and unzipped"