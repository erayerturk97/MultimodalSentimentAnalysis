#!/bin/bash

squeue -u yciftci | awk '
BEGIN {
    abbrev["R"]="(Running)"
    abbrev["PD"]="(Pending)"
    abbrev["CG"]="(Completing)"
    abbrev["F"]="(Failed)"
}
NR>1 {a[$5]++}
END {
    for (i in a) {
        printf "%-2s %-12s %d\n", i, abbrev[i], a[i]
    }
}'
watch -n 2 squeue -u yciftci --format=\" %all %.10i %.9P %.100j %.8u %.8T %.10M %.9l %.6C %.6D %R\"
