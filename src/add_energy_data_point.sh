#!/bin/bash

job_file=$1
job_info_file=$2
raw_energy=$(cat)

job_cells=$(cut -d ',' -f 1 $job_info_file)
job_repeats=$(cut -d ',' -f 2 $job_info_file)
rm $job_info_file

# Any energy value above 2^48 Joules is considered as an overflow.
# We must use bc here, since bash only handles Int64, but we need at least UInt64.
overflow=$(echo "$raw_energy > 2^48" | bc)

if [ "$overflow" -eq 1 ]; then
    # TODO: 104403822 is a magic constant which might (must) need some adusting
    energy=$(echo "($raw_energy + 104403822) % (2^64)" | bc)
    echo "Energy counter overflow: $raw_energy, correcting to: $energy"
else
    energy=$raw_energy
fi

echo "Used $energy Joules for $job_cells cells over $job_repeats repeats"
echo "$job_cells, $energy, $job_repeats" >> $job_file
