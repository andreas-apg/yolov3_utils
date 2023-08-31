#!/bin/bash

for date in exemplo
do
    find $date -maxdepth 1 -type f |
    while IFS= read file_name; do
        python night_sky_higher_gamma.py -i "$file_name"
    done
done

