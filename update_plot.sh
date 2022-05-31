#!/bin/bash
set -e

display_cmd="gio open"
last_update_file_path=./plots/last_update

# Set initial time of file
if [ -e $last_update_file_path ]
then
    last_update_timestamp=`stat -c %Z $last_update_file_path`
else
    last_update_timestamp=0
    touch $last_update_file_path
fi

# If the timestamp changes, read the file to get the path to the updated plot and update its display
while true    
do
   file_timestamp=`stat -c %Z $last_update_file_path`

   if [[ "$file_timestamp" != "$last_update_timestamp" ]]
   then    
       last_update_timestamp=$file_timestamp
       updated_plot=`echo $(<$last_update_file_path)`
       #echo Updated plot '$updated_plot'
       $display_cmd $updated_plot 2> /dev/null
   fi
   sleep 0.1
done
