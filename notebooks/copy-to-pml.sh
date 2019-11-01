#!/bin/bash

PML_DIR=../../intro-to-ml

if [ ! -d $PML_DIR ]
then
    echo "ERROR: could not find PML_DIR=$PML_DIR"
    exit 1
fi

(cd $PML_DIR
 GIT_STATUS=$(git status --porcelain)
 if [ "$GIT_STATUS" != "" ]
 then
     echo "$PML_DIR not up-to-date with git!"
     exit 1
 fi
 git pull
)

try_copy () {
    FROM=$1
    TO=${PML_DIR}/$2
    ls -l $FROM
    ls -l $TO
    cp -iv $FROM $TO
}

try_copy sklearn-mnist-lc.ipynb Exercise-02.ipynb
try_copy sklearn-mnist-svm.ipynb Exercise-05.ipynb
try_copy tf2-mnist-mlp.ipynb Exercise-09.ipynb
try_copy sklearn-mnist-nb.ipynb Extra-01.ipynb
try_copy sklearn-mnist-grid.ipynb Extra-02.ipynb
