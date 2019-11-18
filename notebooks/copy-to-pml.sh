#!/bin/bash

PML_DIR=../../intro-to-ml

if [ ! -d $PML_DIR ]
then
    echo "ERROR: could not find PML_DIR=$PML_DIR"
    exit 1
fi

(cd $PML_DIR
 GIT_STATUS=$(git status --porcelain -uno)
 if [ "$GIT_STATUS" != "" ]
 then
     echo "$PML_DIR not up-to-date with git!"
     exit 1
 fi
 git pull
)

try_copy () {
    echo ""
    FROM=$1
    TO=${PML_DIR}/$2
    ls -l $FROM
    if [ -f $TO ]
    then
        ls -l $TO
    fi
    if cmp -s $FROM $TO
    then
        echo "Files $FROM and $TO are identical"
    else
        cp -iv $FROM $TO
    fi
}

try_copy pml_utils.py             pml_utils.py
try_copy sklearn-mnist-lc.ipynb   Exercise-02.ipynb
try_copy sklearn-mnist-nn.ipynb   Exercise-03.ipynb
try_copy sklearn-chd-lr.ipynb     Exercise-04.ipynb
try_copy sklearn-mnist-svm.ipynb  Exercise-05.ipynb
try_copy sklearn-chd-svm.ipynb    Exercise-06.ipynb
try_copy sklearn-mnist-dt.ipynb   Exercise-07.ipynb
try_copy sklearn-chd-dt.ipynb     Exercise-08.ipynb
try_copy tf2-mnist-mlp.ipynb      Exercise-09.ipynb
try_copy tf2-chd-mlp.ipynb        Exercise-10.ipynb
try_copy sklearn-mnist-dr.ipynb   Exercise-11.ipynb
try_copy sklearn-mnist-viz.ipynb  Exercise-12.ipynb
try_copy sklearn-mnist-cl.ipynb   Exercise-13.ipynb
try_copy sklearn-mnist-ad.ipynb   Exercise-14.ipynb
try_copy sklearn-mnist-nb.ipynb   Extra-01.ipynb
try_copy sklearn-mnist-grid.ipynb Extra-02.ipynb
try_copy sklearn-mnist-ens.ipynb  Extra-03.ipynb
