#!/bin/bash

shopt -s extglob

CASE="matSurr"

FILELIST="
.gitignore
$CASE.dat
$CASE.dom.dat
$CASE.geo
$CASE.ker.dat
$CASE.nsi.dat
$CASE.post.alyadat
$CASE.msh
clean.sh
"

IGNORE=""

i=1
for f in $FILELIST; do
    if [ $i -ne 1 ]; then
        IGNORE+="|"
    fi
    IGNORE+="${f}"
    i=$((i + 1))
done

COMM="rm -rv !($IGNORE)"

echo "$COMM"
eval "$COMM"
