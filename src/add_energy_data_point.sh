#!/bin/bash

job_file=$1
job_cells=$2
STDIN=$(cat)

echo "$2, $STDIN" >> $job_file
