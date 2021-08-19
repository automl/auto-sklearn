# Contributing to auto-sklearn
Thanks for checking out the contribution guide! 
We included a quick overview at the start for anyone familiar with open-source contribution or autosklearn and simply wants to see our workflow.

If you're new to contributing then hopefully this guide helps with contributing to open-source and auto-sklearn, whether it be a simple doc fix, a small bug fix or even new features that everyone can get use out of.
Even if you're new to open-source or even new to Python but you want to get your hands dirty, we'll try to be as helpful as possible.
If you're looking for a particular project to work on, check out the [Issues](https://github.com/automl/auto-sklearn/issues) for things you might be interested in!

For new contributors, you can skip the overview and find the more detailed walkthrough [here](#contributing)!

This guide is only aimed towards Unix command line users as that's what we know but the same principles apply.

# Pull Request Overview
* Create a fork of the [automl/auto-sklearn](https://github.com/automl/auto-sklearn) git repo
* Clone your own fork and create a new branch from the branch to work on
    ```bash
    git clone git@github.com:automl/auto-sklearn.git
    cd auto-sklearn
    git checkout -b my_new_branch development
    
    python -m venv my-virtual-env
    source my-virtual-env/bin/activate
    
    pip install -e .[test,docs,examples] # zsh users need quotes ".[test,...]"
    
    # Edit files...
    
    # If you changed documentation:
    # This will generate all documentation, run examples and check links
    cd doc
    make linkcheck
    
    # ... fix any issues
    
    # If you edited any code
    # Check out pytest --help if you want to only run specific tests
    pytest
    
    # ... fix any issues
    
    # Use pre-commit for style and typing checks
    pip install pre-commit
    pre-commit run --all-files
    
    # ... fix any issues
    
    # Add the changes
    git add {changed files}
    git commit -m "Meaningful as you can make it message"
    
    # Push back to your fork
    git push --set-upstream origin my_new_branch
    ```
* Go to github, go to our own fork and then make a pull request using the *Contribute*
icon.
    * `automl/auto-sklearn` | `development` <- `your-username/auto-sklearn` | `my_new_branch`
* Write a description of the changes, why you implemented them and any implications.
* If it's your first time contributing, we will run some automated tests, mostly
the same as you can run manually
* We'll review the code and perhaps ask for some changes
* Once we're happy with the result, we'll merge it in!

# Contributing
There are many kinds of contributions you can make to auto-sklearn but we'll focus on three main ones **Documentation**, **Bug Fixes** and **Features**, each of which require a little bit of a different flow to make sure it meets code standards and won't cause any issues later on.

First we'll go over the general flow, what each step does and then later look at making more specific kinds of changes, what we'd like to see and how you might create a workflow.

## General steps
*   The first thing to do is create your own [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
    This gives you a nice place to work on your changes without impacting any code from the original repository.
    To do this, navigate to [automl/auto-sklearn](https://github.com/automl/auto-sklearn) and hit the **fork** button in the top-right corner.
    This will copy the repository as it is, including all its different branches, to you own account.
    You'll be able to access this at `https://github.com/{your-username}/auto-sklearn`.

*   The next steps are to download **your own fork** and to create a new [branch](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches) where all your changes will go.
    ```bash
    # With https
    git clone https://github.com/{your-username}/auto-sklearn
    
    # ... or With ssh
    git clone git@github.com:automl/auto-sklearn.git
    
    # Navigate into the cloned repo
    cd auto-sklearn
    
    # Switch to the development branch
    git checkout development
    
    # Show all branches and also a * beside the one you are on
    git branch
    
    # Create a new branch based off the development one
    git checkout -b my_new_branch
    ```
    The reason to create a new branch is two fold
    
    *   It lets us keep the commit history cleaner once you want to make a pull
    request.
    *   If you have to perform a **rebase** or a **merge** later on, this will be
    much easier.
        
*   You'll need a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to work in.
    If you've never used them before, now is definitely the time to start as a virtual environment lets you keep packages for a project separate.
    ```bash
    # Create a virtual environment in a folder called my-virtual-env
    python -m venv my-virtual-env
    
    # Activate the virtual environment
    source my-virtual-env/bin/activate
    ```
    *   This is the folder where downloaded packages you will need for auto-sklearn will go.
        In general, anything you install with `pip install` will now go into the virtual environment.
        If at any time you want to deactivate it simply type `deactivate` in the shell or just close the shell and open a new one
    *   As extra steps, if you use Python 3.6 or lower on your machine, unfortunately auto-sklearn doesn't support this.
        You can check out [pyenv](https://github.com/pyenv/pyenv) which lets you switch between Python versions on the fly and manages them all for you
    
*   Now that we have a virtual environment, it's time to install all the dependencies into it.
    ```bash
    pip install -e .[test,examples,doc]
    
    # If you're using shells other than bash you'll need to use
    pip install -e ".[test,examples,doc]"
    ```
    *   If you're only use to using `pip install package_name` then this might be a bit confusing.
    *   If we type `pip install -e .` (notice the 'dot'), this tells `pip` to install a local package located here, in this directory, `.`. 
        The `-e` flag indicates that it should be editable.
        This means that you want have to run `pip install .` every time you make a change and want to use it.
    *   Finally the `[test,examples,doc]` tells `pip` that there's some extra optional dependencies that we want to install.
        These are used in development but dependencies that aren't required to actually run autosklearn itself.
        You can check out what these are in the `setup.py` file!
    *   If you new to virtual environments, this is a great time to check out what actually exists in the `my-virtual-env`.

*   Now it's time to make some changes, whether it be for [documentation](#documentation), a [bug fix](#bug-fixes) or a new [features](#features).

*   Let's assume you've made some changes, now we have to make sure they work.
    Begin by simply running all the tests.
    If there's any errors, they'll pop up once it's complete.
    ```bash
    pytest
    ```
    *   Note that these may take a while so check out `pytest --help` to see how you can run tests so that only previous failures run or only certain tests are run.
        This can help you try changes and get results faster.
        Do however run one last full `pytest` once you are finished and happy!

*   Now we are going to use [sphinx](https://www.sphinx-doc.org/en/master/) to generate all the documentation and make sure there are no issues.
    ```bash
    cd doc
    make html
    ```
    *   If you're unfamiliar with sphinx, it's a documentation generator which can read comments and docstrings from within the code and generate html documentation.
    *   We also use sphinx-gallery which can take python files (such as those in the `examples` folder) and run them, creating html which shows the code and the output it generates.
        Unfortunately this can take quite some time but you should only have to run this once.
    *   If you've made many documentation changes, there are more explicit details for [testing documentation changes](#documentation)

*   Once you've made all your changes and all the tests pass successfully, we need to make sure that the code fits a certain format and that the [typing](https://docs.python.org/3/library/typing.html) is correct.
    To do this, we use a tool call `pre-commit` which runs `flake8`, a code checker and `mypy`, a static type checker against the code.
    ```bash
    pip install pre-commit
    pre-commit run --all-files
    ```
    *   The reason we use a code standard (e.g. `flake8` )is to make sure that when we review code:
        *   There are no extra blank spaces and blank lines.
        *   Lines don't end up too long
        *   Code from multiple source keeps a similar appearance.
    *   We perform static type checking with `mypy` as this can remove a majority of bugs, before a test is even run.
        It points out programmer errors and what makes compiled languages so safe so that is why we try to use it as much as possible.
        If you are new to Python types, or stuck with how something should be 'typed', please feel free to push the pull request in the following steps and we should be able to help you out.
    * If interested, the configuration for `pre-commit` can be found in `.pre-commit-config.yaml`

*   Finally, we should commit our changes, push them up to our fork and create a pull request.
    ```bash
    # Add your changes and make a commit
    git add {changed files}
    git commit -m "Something as meaningful as possible"
    
    # This will push my_new_branch to your fork located at `origin`
    git push --set-upstream origin my_new_branch
    ```

*   At this point, we need to create a PR request to the [automl/autosklearn](https://github.com/automl/auto-sklearn) repository with our new changes.
    This can be done simply by going to your own forked repo, clicking **'Contribute'**, and selecting
    the **development** branch of `automl/auto-sklearn`.
    *   `automl/auto-sklearn` | `development` <- `your-username/auto-sklearn` | `my_new_branch`
    The reason we don't want to directly merge new PR's into master is to make
    sure we always have a stable version. With a development branch, we can safely
    accumulate certain changes and makes sure they all work together before creating
    a new master version.

*   Now you've got to describe what you've changed.
    Some key things to include here are:
    *   A high level overview of what you've done such as fixing a problem or
        introducing a new feature.
        If it's a simple doc fix, don't worry too much about this.
    *   Have you introduced any breaking changes to the API?
        We may not realise it while reviewing the changes but if you are aware,
        it definitely helps to tell us!
    *   Do you think this might have any implications in the future or how would
        further work on this look like?
    *   If you've introduced some new feature, write about who might use it,
        give a breif code sample to show it and let us know how you tested it!
    *   If you've fixed a bug, write about why this bug exists in the first place,
        how you solved it and how a test makes sure it won't pop up again!

*   Once you've submitted a PR, we'll probably enable some automatic tests to run.
    This will make sure all the tests run smoothly, make sure the documentation builds correctly and do some quick check on code quality.
    You'll be able to see these run in the **Checks** tab or at the bottom of the PR.
    If you see a red x, that means somethings probably gone wrong which should have been caught by running the tests locally but we also do some checks in environments you were not developing on.
    Sometimes it is the case that there is a bug unrelated to your changes that cause the tests to fail but we are aware of these.
    If you see one of these failures, feel free to ask!

*   Meanwhile, we'll also review your code.
    In the top right you'll see a little **Suggested Reviewers** button so feel free to request a review with that!
    Some common review points are:
    * This seems odd, why was this done?
    * Could you see if you can use the functionality from place X?
    * Could you creat a test or documentation for this?
    * This could be a nicer way of doing this, what do you think?
    Occasionally there will be some major point which will require more discussion but those are more on a case-by-case basis.

*   This process of review, fix and testing may go on a few times.
    The simplest way to reduce the time needed and to help us too is to run the tests, code formatting check and doc building locall.
    If they all pass locally they will very often have no issues in the automated tests.

    Once everyone is happy, it's time for us to hit (*squash*) merge, get it into the development branch and have one more contribution to autosklearn!

## Kinds of PRs
Of course we can't cover every use case but here's some common kinds of PR's along with some pointers!

#### Documentation
Anything to contribute to better documentation is always appreciated
#### Bug Fixes
* TODO
#### Features
* TODO

# General FAQ

### I've finished my pull request or made new changes and want it reviewed, now what?
* In the Pull request on the right there should be a tab **Reviewers** with some suggested
reviewers, feel free to request once you think your contribution is ready.

### I've been asked to rebase my pull request, why and what do I do?
* It can often be the case that while you were working on some changes, we may have merged something new into the development branch.
This is a problem because the branch you created was based off the old development branch in and so there are no changes in the automl/auto-sklearn code base you don't have in your branch.
This is not always an issue, as generally if different parts of the code were touched, they can be merged safely.
Either way, if we ask you to [rebase](https://www.atlassian.com/git/tutorials/merging-vs-rebasing), it is because it won't merge or we think there may have been overlapping changes between what you were working on and the new code put into the development branch.
* First, update the development branch on **your fork**
    * The easiest way to do this is go to github.com/{your-username}/auto-sklearn navigate to the development branch and click *fetch upstream*.
    * This will make it so your fork of auto-sklearn is now up to date
* Second, we need to go to our clone and pull in these new changes to development
    ```bash
    git checkout development
    git pull
    ```
* Lastly, we need to rebase the my_new_branch on top of the new development
    ```bash
    git checkout my_new_branch
    git rebase development
    ```
* Now if there were no conflicts, that's it, you can continue as normal. If there
are conflicts, you'll have to sort these out which you can find out how to do
[here](https://docs.github.com/en/get-started/using-git/resolving-merge-conflicts-after-a-git-rebase) and [here](https://docs.github.com/en/github/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line).

