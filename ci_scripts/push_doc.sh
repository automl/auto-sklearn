#!/bin/bash
# This script is meant to be called in the "deploy" step defined in 
# circle.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the circle.yml in the top level folder of the project.

if [ ! -z "$1" ]
    then DOC_FOLDER=$1
fi

MSG="Pushing the docs for revision for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1, folder: $DOC_FOLDER"

cd $HOME

# Clone the docs repo if it isnt already there
if [ ! -d $DOC_REPO ];
    then git clone "git@github.com:$USERNAME/"$DOC_REPO".git";
fi

# Copy the build docs to a temporary folder
rm -rf tmp
mkdir tmp
cp -R $HOME/$DOC_REPO/doc/build/html/* ./tmp/

cd $DOC_REPO
git branch gh-pages
git checkout -f gh-pages
git reset --hard origin/gh-pages
git clean -dfx
git rm -rf $HOME/$DOC_REPO/$DOC_FOLDER && rm -rf $HOME/$DOC_REPO/$DOC_FOLDER

# Copy the new build docs
mkdir $DOC_FOLDER
cp -R $HOME/tmp/* ./$DOC_FOLDER/

git config --global user.email $EMAIL
git config --global user.name $USERNAME
git add -f ./$DOC_FOLDER/
git commit -m "$MSG"
git push -f origin gh-pages

echo $MSG