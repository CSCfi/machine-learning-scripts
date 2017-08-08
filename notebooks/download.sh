#!/usr/bin/env bash

## Step 1. Install needed Python packages:

pip install --user sklearn xgboost bokeh lasagne graphviz scikit-image

## Step 2. Download notebooks and misc stuff from Github:

base="https://raw.githubusercontent.com/CSCfi/machine-learning-scripts/keras1-legacy/notebooks"

for i in keras-test-setup.ipynb \
         keras-mnist-mlp.ipynb \
         keras-mnist-cnn.ipynb \
         sklearn-mnist-dt.ipynb \
         sklearn-mnist-nn.ipynb \
         sklearn-mnist-svm.ipynb \
         sklearn-mnist-viz.ipynb \
	 sklearn-mnist-dr.ipynb
do
    wget "$base/$i"
done

mkdir imgs && cd imgs

for i in 500px-KnnClassification.svg.png \
         Svm_max_sep_hyperplane_with_margin_small.png \
         dtree.png
do
    wget "$base/imgs/$i"
done

## All done.
