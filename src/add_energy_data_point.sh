#!/bin/bash

job_file=$1
job_cells=$2
STDIN=$(cat)

# We must use bc here, since bash only handles Int64, but we need at least UInt64
overflow=$(echo "$STDIN > 2^64" | bc)

if [ "$overflow" -eq 1 ]; then
    # TODO: 104403822 is a magic constant which might (must) need some adusting
    energy=$(echo "($STDIN + 104403822) % (2^64)" | bc)
    echo "Energy counter overflow: $STDIN, correcting to: $energy"
else
    energy=$STDIN
fi

echo "Used $energy Joules for $job_cells cells"
echo "$job_cells, $energy" >> $job_file
