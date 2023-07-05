#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

session="job-monitor"

tmux new-session -d -s $session
window=0

while IFS=$'\t' read -r job_id job_stdout job_stderr
do
    tmux rename-window -t $session:$window "$job_id"
    tmux select-window -t $session:$window
    tmux split-window -v

    tmux select-pane -t 0
    tmux send-keys -t $session:$window "tail -f $job_stdout" Enter

    tmux select-pane -t 1
    tmux send-keys -t $session:$window "tail -f $job_stderr" Enter

    window=$((window+1))
    tmux new-window -t $session

done < "$SCRIPT_DIR/recent_jobs.txt"

tmux send-keys -t $session:$window "gio open $job_stderr"

tmux select-window -t $session:0
tmux attach-session -t $session
