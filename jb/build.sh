#!/bin/bash
# Build the Jupyter Book site; with "publish", also push it to the gh-pages branch.
#   cd jb; ./build.sh [publish]

set -e

python prep_notebooks.py
jb build .

if [ "$1" == "publish" ]; then
    ghp-import -n -p -f _build/html
fi
