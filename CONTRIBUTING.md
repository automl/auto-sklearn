# Contributing to auto-sklearn
Thanks for checking out the contribution guide! Feel free to [skip ahead](#pull-request-overview) if you're not new to open-source or already contributed before. 

Hopefully this guide helps you to contribute to open-source and auto-sklearn, whether it be a simple doc fix, a small bug fix or even new features that everyone can get use out of.
If you're new to open-source or even new to Python but you want to get your hands dirty, we'll try to be as helpful as possible.
If you're looking for a particular project to work on, check out the [Issues](https://github.com/automl/auto-sklearn/issues) for things you might be interested in working on!

For new contributors, the first sections will be of most help while contributors to other open-source projects or even auto-sklearn can refer to the [details](#pull-request-overview) section.

This guide is only aimed towards Unix command line users as that's what we know but the same principles apply.

# Contributing Overview
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

* Creating a PR TODO
* Writing a PR description TODO
* Automated tests TODO
* Fixing Review changes TODO

## Documentation
* TODO
## Bug Fixes
* TODO
## Features
* TODO

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

# FAQ

### I've finished my pull request and want it reviewed, now what?
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

