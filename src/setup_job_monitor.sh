#!/bin/bash

# From a Slurm job ID list, create as many tmux windows with two panes: one for stdout, another for stderr

if [[ "$#" -eq 0 ]]; then
    # read job list from ./recent_jobs.txt
    SCRIPT=$(readlink -f "$0")
    SCRIPT_DIR=$(dirname "$SCRIPT")
    job_list=$(cat "$SCRIPT_DIR/recent_jobs.txt")
elif [[ "$#" -eq 1 ]]; then
    if [[ -f "$1" ]]; then
        # read job list from file
        job_list=$(cat $1)
    else
        job_list="$1"
    fi
else
    # read job list from args
    job_list="$@"
fi

session="job-monitor"

# New session, or add a new window if the session already exists
window_count=$(tmux list-windows -t $session 2&> /dev/null | wc -l)
if [[ "$window_count" -ge 1 ]]; then
    window=$window_count
    tmux new-window -t $session
else
    tmux new-session -A -d -s $session
    window=0
fi
init_window=$window

for job_id in $job_list; do
    # Use `scontrol` to get the output files of the job (and replace %J by the job id)
    job_info=$(scontrol show jobid=$job_id)
    job_stdout=$(echo $job_info | grep -Po "(?<=StdOut=).*$" | awk '{ gsub("%J",'$job_id',$1); print $1 }')
    job_stderr=$(echo $job_info | grep -Po "(?<=StdErr=).*$" | awk '{ gsub("%J",'$job_id',$1); print $1 }')

    tmux rename-window -t $session:$window "$job_id"
    tmux select-window -t $session:$window

    tmux split-window -v

    tmux select-pane -t 0
    tmux send-keys -t $session:$window "tail -f $job_stdout" Enter

    tmux select-pane -t 1
    tmux send-keys -t $session:$window "tail -f $job_stderr" Enter

    window=$((window+1))
    tmux new-window -t $session
done

tmux select-window -t $session:$init_window
tmux attach-session -t $session
