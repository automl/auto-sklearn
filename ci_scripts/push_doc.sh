#!/bin/bash
# This script is meant to be called in the "deploy" step defined in 
# circle.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the circle.yml in the top level folder of the project.

MSG="Pushing the docs for revision for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

cd $HOME
# Copy the build docs to a temporary folder
rm -rf tmp
mkdir tmp
cp -R $HOME/$DOC_REPO/doc/build/html/* ./tmp/

# Clone the docs repo if it isnt already there
if [ ! -d $DOC_REPO ];
    then git clone "git@github.com:$USERNAME/"$DOC_REPO".git";
fi

cd $DOC_REPO
git branch gh-pages
git checkout -f gh-pages
git reset --hard origin/gh-pages
git clean -dfx

for name in $(ls -A $HOME/$DOC_REPO); do
    case $name in
        .nojekyll) # So that github does not build this as a Jekyll website.
        ;;
        circle.yml) # Config so that build gh-pages branch.
        ;;
        *)
        git rm -rf $name
        ;;
    esac
done

# Copy the new build docs
mkdir $DOC_URL
cp -R $HOME/tmp/* ./$DOC_URL/

git config --global user.email $EMAIL
git config --global user.name $USERNAME
git add -f ./$DOC_URL/
git commit -m "$MSG"
git push -f origin gh-pages

echo $MSG 
