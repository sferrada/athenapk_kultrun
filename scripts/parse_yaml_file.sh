#!/bin/bash

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  "$1" |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         gsub(/#[^"]*/, "", $3)  # Remove comments
         # Trim any leading or trailing spaces from the variable value
         value = gensub(/^[[:space:]]+|[[:space:]]+$/, "", "g", $3)
         # Handle array values
         if (value ~ /^\[.*\]$/) {
            split(value, arr, /[\[\],]+/)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", arr[1])
            for (i = 2; i <= length(arr); i++) {
               if (arr[i] != "") {
                  printf("%s%s%s+=(\"%s\")\n", "'$prefix'", vn, $2, arr[i])
               }
            }
         } else {
            printf("%s%s%s=\"%s\"\n", "'$prefix'", vn, $2, value);
         }
      }
   }'
}
