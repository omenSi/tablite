#!/bin/bash

if [[ $# -eq 0 ]]
then
    is_release=true
elif [[ $# -eq 1 ]]
then
    if [[ "$1" == "--debug" ]]
    then
        is_release=false
    else
        is_release=true
    fi
    
else
    echo "Invalid arguments '$@'"
    exit 1
fi

if [ $is_release = true ]
then
    nim c --app:lib --gc:refc -d:release -d:danger --out:nimlite/libnimlite.so nimlite/libnimlite.nim
    echo "Built release."
else
    nim c --app:lib --gc:refc -d:debug --out:nimlite/libnimlite.so nimlite/libnimlite.nim
    echo "Built debug."
fi