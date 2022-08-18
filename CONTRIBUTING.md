# Contributing to auto-sklearn
Thanks for checking out the contribution guide!
We included a [quick overview](#pull-request-overview) at the end for anyone familiar with open-source contribution or familiar with auto-sklearn and simply wants to see our workflow.

If you're new to contributing then hopefully this guide helps with contributing to open-source and auto-sklearn, whether it be a simple doc fix, a small bug fix or even new features that everyone can get use out of.
If you're looking for a particular project to work on, check out the [Issues](https://github.com/automl/auto-sklearn/issues) for things you might be interested in!

For experienced contributors, you can skip the overview and find the quick walk-through [here](#pull-request-overview)!

This guide is only aimed towards Unix command line users as that's what we know but the same principles apply.

# Contributing
There are many kinds of contributions you can make to auto-sklearn but we'll focus on three main ones **Documentation**, **Bug Fixes** and **Features**, each of which require a little bit of a different flow.
We need to perform several checks to make sure it meets code standards and won't cause any issues later on.

We tend to follow a development cycle which could be called _Gitflow_ which you are new to git or git-based projects, you can see a nice summary [here](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

First we'll go over the general flow, what each step does and then later look at making more specific kinds of changes, what we'd like to see and how you might create a workflow.
Following that we'll tell you about how you can test your changes locally and then how to submit your pull request!

## General steps
*   The first thing to do is create your own [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
    This is to give you a nice place to work on your changes without impacting any code from the original repository.

    To do this, navigate to [automl/auto-sklearn](https://github.com/automl/auto-sklearn) and hit the **fork** button in the top-right corner.
    This will copy the repository to your own account, including all of its different branches.
    You'll be able to access this at `https://github.com/{your-username}/auto-sklearn`.

*   The next steps are to download **your own fork** and to create a new [branch](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches) where all your changes will go.
    It's important to work off the latest changes on the **development** branch.
    ```bash
    # With https
    # Note the --recurse-submodules args, we use a submodule autosklearn/automl_common
    # so it needs to be downloaded too
    git clone --recurse-submodules https://github.com/your-username/auto-sklearn

    # ... or with ssh
    git clone --recurse-submodules git@github.com:your-username/auto-sklearn.git

    # Navigate into the cloned repo
    cd auto-sklearn

    # Create a new branch based off the development one
    git checkout -b my_new_branch development

    # If you missed the --recurse-submodules arg during clone or need to install the
    # submodule manually, then execute the following line:
    #
    # git submodule udate --init --recursive

    # ... Alternatively, if you would prefer a more manual method
    # Show all the available branches with a * beside your current one
    git branch

    # Switch to the development branch
    git checkout development

    # Create a new branch based on the currently active branch
    git checkout -b my_new_branch

    # If you missed the --recurse-submodules arg during clone or need to install the
    # submodule manually, then execute the following line:
    #
    # git submodule update --init --recursive
    ```

    The reason to create a new branch is two fold:

    *   One, it keeps the commit history for your changes much cleaner once we merge them in.
    *   If you have to perform a **rebase** or a **merge** later on, this will be much easier.

*   You'll need a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to work in.
    If you've never used them before, now is definitely the time to start as a virtual environment lets you keep packages for a project separate.
    ```bash
    # Create a virtual environment in a folder called my-virtual-env
    python -m venv my-virtual-env

    # Activate the virtual environment
    source my-virtual-env/bin/activate
    ```
    *   A popular alternative to managing Python projects is [conda](https://docs.conda.io/en/latest/).
    *   The folder `my-virtual-env` is where dependency packages that are required for auto-sklearn will go.
    *   In general, once you have activated the virtual environment with `source my-virtual-env/bin/activate`, anything you install with `pip install` will now go into the virtual environment.
        While this environment is active, any python you run will have access to the packages here.
        If at any time you want to deactivate it simply type `deactivate` in the shell or just close the shell and open a new one.
    *   If you use Python 3.6 or lower on your machine then unfortunately auto-sklearn doesn't support this.
        Fortunately, you can check out [pyenv](https://github.com/pyenv/pyenv) which lets you switch between Python versions on the fly!

*   Now that we have a virtual environment, it's time to install all the dependencies into it.
    We've provided a simple `make` command to help do this.
    ```bash
    make install-dev

    # Manually
    pip install -e .[test,examples,doc]

    # If you're using shells other than bash you'll need to use
    pip install -e ".[test,examples,doc]"
    ```
    *   If your only exposure to using pip is `pip install package_name` then this might be a bit confusing.
    *   If we type `pip install -e .` (notice the 'dot'), this tells `pip` to install a package located here, in this directory, `.`.
        The `-e` flag indicates that it should be editable, meaning you will not have to run `pip install .` every time you make a change and want to try it.
    *   Finally the `[test,examples,doc]` tells `pip` that there's some extra optional dependencies that we want to install.
        These are dependencies used in development but ones that are not required to actually run auto-sklearn itself.
        You can check out what these are in the `setup.py` file.
    *   If you're new to virtual environments, after performing all this, it's a great time to check out what actually exists in the `my-virtual-env` folder.

* You can check out some functionality we have captured in a `Makefile` by running `make help`

*   Now it's time to make some changes, whether it be for [documentation](#documentation), a [bug fix](#bug-fixes) or a new [features](#features).

## Making Changes
We'll go over three main categories of contributions but don't feel limited by these headers, adding to our tests, improving the typing of functions and methods or even some compliance changes are also super useful!

#### Bug Fixes
Auto-sklearn has been through quite a few iterations, been used for many purposes and is constantly used in ways we didn't even think of.
Like any maturing software, there will be bugs, old and new.
While we try to create unit tests to catch as many of these as we can, some slip through and new bugs get introduced as dependencies are updated and new features are introduced.

If you're looking to help by fixing some bugs, or you've encountered your own bugs you'd like fixed in the official version of auto-sklearn, we would greatly appreciate any help!
We'd be happy to guide you through what we think may be the underlying cause or at least point you in the right direction if it's something you wish to work on.

Some core things to consider in fixing a bug:
*   What's the minimal working example that reproduces this bug?
    *   This is usually the first step.
        The process of creating a minimal piece of code that reproduces a bug often illuminates what needs to be fixed.

*   What's the quick fix and what's the long term fix?
    *   Sometimes it's a code typo, a quick correction and problem solved.
        Other times, the bug is an artifact of some larger underlying issue that has gone unnoticed and might require some restructuring.

        If this is the case, let us know as you work on it!
        If it requires breaking, such as changing default behaviour or public API, sometimes a quick patch and fix will do and larger restricting fixes can be tackled in a timely manner.
        As a rule of thumb, if a bug requires modifying more then 50-100 lines of code it's probably something we would like to talk through on how best to tackle it.

*   How can we create a test for this bug in the future?
    *   Of course, once a bug is squashed, we'd like it to not show again and having a test to catch it for the future will future-proof against any changes down the line.

        Thankfully, most of this is usually captured in the minimal working example and all that is left is to turn it into a comprehensive test!

What's important once fixing a bug is [writing a good PR](#creating-the-pr) that let's us now how you identified the bug, what the problem was and how it was fixed.
This lets use review your code with all this in mind and follow the same thought process that lead you to fix it in the first place!

#### Documentation
Anything to contribute to better documentation is always appreciated and the main way users can get to know about auto-sklearn.
Whether it's a typo fix, something you didn't find clear or something you think we didn't explain properly, we'd love to improve it!

All of our documentation is done with [`sphinx`](https://www.sphinx-doc.org/en/master/) with some various
plugins that you can see in [`doc/conf.py`](https://github.com/automl/auto-sklearn/blob/master/doc/conf.py#L42).

All of your changes can be viewed by first [building the docs](#testing) and then opening `doc/build/html/index.html` in a browser.

*   If you're simply fixing a typo, there shouldn't be much to do except make the PR and we'll accept it without much issue.
*   If you want to fix a link you should know how linking with `sphinx` works
    *   For links to internal documentation, you can create a label with
    ```.rst
    .. _mylabel:
    ```
    *   Later on you can reference this label by
    ```.rst
    I am reference to :ref:`the above label<mylabel>`
    ```
    *   Now if you want to link some external documentation you'll need
        to do something like
    ```.rst
    Here's a `link<https://sublime-and-sphinx-guide.readthedocs.io/en/latest/references.html>`_ to the external documentation on linking for sphinx
    ```
    Notice the trailing `_` which is important

*   If you want to make some more detailed documentation about some feature that you introduced or you think is not well documented, you'll have to think about a few things.
    *   Can you include a code snippet to illustrate what you mean?
    *   Are there other relevant parts of the documentation or code that should be linked to?
    *   How much other parts of auto-sklearn are you relying on readers to know before hand, maybe link to those sections if you do.

*   If you want to contribute an example, it's a great way to really illustrate an entire flow of some feature.
    `sphinx-gallery` will run any python file `example_*.py` in one of the example folders.
    This allows you to have both ReStructured Markdown (rst) and python code with it's output into a [single html page](https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_calc_multiple_metrics.html#sphx-glr-examples-40-advanced-example-calc-multiple-metrics-py)!
    You'll want to check out some of the [other examples](https://github.com/automl/auto-sklearn/tree/master/examples) to see how to embed rst into a `.py` file.

#### Features
While auto-sklearn has many features we're proud of, there's always room for more and better functionality.
Features don't have to be performance driven, in fact, most of the new features we'd love to see are to improve a users ability to interact with auto-sklearn, whether it be usability or an ability to inspect the inner workings in an intuitive manner!

However, features are usually a bigger project and for this we'd really advise getting in touch with us first about the feature in mind.
There are some things we believe best left for external libraries or integrations that we don't wish to consider at this time.
Another reason to get in touch is to nail exactly what this feature will look like beforehand, the more direct and concise the feature is, the better.

Some things to keep in mind with new features:
*   A new feature is great, but will this change any existing default behaviours?
    Unexpected changes for existing users can be detrimental and unexpected.
    Sometimes it has to be done but often these new features can be presented as an option the user can enable.
*   If you're introducing some new API:
    *   Creating some code samples of how you'd like your feature to be used is a great start.
    *   What current way is there to do the same thing, can any functionality already present be used to help with this new API?
    *   Are you going to deprecate any current API? This is an important fact to consider and something that will definitely have to be discussed.

**Testing Features** - Writing features are great but new features means new bugs, but thankfully that's what we can write tests for.
How to [test your feature](#testing) is always tricky, especially if the feature is big in scope.
Unfortunately there's no secret or rule of thumb other than try to cover every case you can think of.
The more it's tested the better!
Bugs will still get through, that's okay, we will have done what we can and we can fix those in the future but as long as the usual use-cases are covered, this shouldn't be too much of a problem.

**Documenting Features** - Now how are people going to know about your new feature you've introduced?
This is what [documentation](#documentation) is so great for and it's how almost all software functionality is expressed.
If the feature is enabled by a parameter, great, almost all the documentation is already present in the code docstring, automatically being rendered in the online docs ... that docstring that was updated when you made changes ... right?
Sometimes, the new functionality isn't so clear from a simple parameter description and so maybe something needs to be added to the `manual.rst`, a short paragraph suffices and much appreciated.
Lastly, if the feature really is a game changer or you're very proud of it, consider making an `example_*.py` that will be run and rendered in the online docs!

## Testing
* Let's assume you've made some changes, now we have to make sure they work.
    Begin by simply running all the tests.
    If there's any errors, they'll pop up once it's complete.
    ```bash
    pytest
    ```
    * Note that these may take a while so check out `pytest --help` to see how you can run tests so that only previous failures run or only certain tests are run.
        This can help you try changes and get results faster.
        Do however run one last full `pytest` once you are finished and happy!
    * Here are some we find particularly useful
        ```
        # Run tests in specific file like 'test_estimators.py'
        pytest "test/test_automl/test_estimators.py"

        # Run an entire directory of tests such as 'pipeline'
        pytest "test/test_pipeline"

        # Run a specific test 'test_mytest' in a specific directory 'test_automl'
        pytest -k "test_mytest" "test/test_automl"

        # Rerun all the tests that failed in the last `pytest` command
        pytest --last-failed

        # Rerun all tests but run the failed ones first
        pytest --failed-first

        # Exit on the first test failure
        pytest -x
        ```
    * More advanced editors like PyCharm may have built in integrations which could be good to check out!
    * Running all unittests will take a while, here's how you can run them in parallel
        ```
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OMP_NUM_THREADS=1
      
        pytest -n 4
        ```


* Now we are going to use [sphinx](https://www.sphinx-doc.org/en/master/) to generate all the documentation and make sure there are no issues.
    ```bash
    make doc
    ```
    *   If you're unfamiliar with sphinx, it's a documentation generator which can read comments and docstrings from within the code and generate html documentation.
    *   If you've added documentation, we also has a command `linkcheck` for making sure all the links correctly go to some destination.
        This helps tests for dead links or accidental typos.
    ```bash
    make linkcheck
    ```
    *   We also use sphinx-gallery which can take python files (such as those in the `examples` folder) and run them, creating html which shows the code and the output it generates.
    ```bash
    make examples
    ```
    *   To view the documentation itself, make sure it is built with the above commands and then open `doc/build/html/index.html` with your favourite browser:
    ```bash
    # Firefox
    firefox ./doc/build/html/index.html

    # Using your default browser
    xdg-open ./doc/build/html/index.html
    ```

* Once you've made all your changes and all the tests pass successfully, we need to make sure that the code fits a certain format and that the [typing](https://docs.python.org/3/library/typing.html) is correct.
    * Formatting and import sorting can helps keep things uniform across all coding styles. We use [`black`](https://black.readthedocs.io/en/stable/) and [`isort`](https://isort.readthedocs.io/en/latest/) to do this for us. To automatically run these formatters across the code base, just run the following command:
    ```bash
    make format
    ```
    * To then check for issues using [`black`](https://black.readthedocs.io/en/stable/), [`isort`](https://isort.readthedocs.io/en/latest/), [`mypy`](http://mypy-lang.org/), [`flake8`](https://flake8.pycqa.org/en/latest/) and [`pydocstyle`](http://www.pydocstyle.org/en/stable/), run
    ```bash
    make check
    ```
    * To do this checking automatically, we use `pre-commit` which if you already installed everything with `make install-dev` then this has been done for you.
    This will happen every time you make a commit and warn you of any issues.
    Otherwise you can run the following to install pre-commit.
    ```bash
    pre-commit install
    ```
    * To run `pre-commit` manually:
    ```bash
    pre-commit run --all-files
    ```
    *   The reason we use tools like [`flake8`](https://flake8.pycqa.org/en/latest/), [`mypy`](http://mypy-lang.org/), [`black`](https://black.readthedocs.io/en/stable/), [`isort`](https://isort.readthedocs.io/en/latest/) and [`pydocstyle`](http://www.pydocstyle.org/en/stable/) is to make sure that when we review code:
        *   There are no extra blank spaces and blank lines. (`flake8`, `black`)
        *   Lines don't end up too long. (`flake8`, `black`)
        *   Code from multiple source keeps a similar appearance. (`black`)
        *   Importing things is consistently ordered. (`isort`)
        *   Functions are type annotated and correct with static type checking. (`mypy`)
        * Function and classes have docstrings. (`pydocstyle`)
        If you are new to Python types, or stuck with how something should be 'typed', please feel free to push the pull request in the following steps and we should be able to help you out.
    * If interested, the configuration for `pre-commit` can be found in `.pre-commit-config.yaml` with the other tools mainly being configured in `pyproject.toml` and `.flake8`.


## Creating the PR
*   We've made sure all the changes work, we've maybe added a test for them and run all the tests locally.
    It's time to commit the changes, push them up to your fork and create a pull request!
    ```bash
    # Get an overview of all the files changes
    git status

    # Add your changed files
    git add {changed files}
    git commit -m "Something as meaningful as possible"

    # This will push my_new_branch to your fork located at `origin`
    git push --set-upstream origin my_new_branch
    ```

*   At this point, we need to create a pull request (PR) to the [automl/auto-sklearn](https://github.com/automl/auto-sklearn) repository with our new changes.
    This can be done simply by going to your own forked repo, clicking **'Contribute'**, and selecting
    the **development** branch of `automl/auto-sklearn`.

    *   `automl/auto-sklearn` | `development` <- `your-username/auto-sklearn` | `my_new_branch`

        The reason we don't want to directly merge new PR's into master is to make
        sure we always have a stable version. With a development branch, we can safely
        accumulate certain changes and makes sure they all work together before creating
        a new master version.

*   Now you've got to describe what you've changed.
    You'll likely want to check out [this blogpost](https://hugooodias.medium.com/the-anatomy-of-a-perfect-pull-request-567382bb6067) which we believe to give a good overview of what a good PR looks like and will help us get your changes in sooner rather than later!
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
        give a brief code sample to show it and let us know how you tested it!
    *   If you've fixed a bug, write about why this bug exists in the first place,
        how you solved it and how a test makes sure it won't pop up again!

*   Once you've submitted a PR, we have it set up so github will automatically schedule some unit tests and documentation building to run.
    This will make sure all the tests run smoothly, make sure the documentation builds correctly and do some quick check on code quality.
    You'll be able to see these run in the **Checks** tab or at the bottom of the PR.
    If you see a red x, that means somethings probably gone wrong which should have been caught by running the tests locally but we also do some checks in environments you were not developing on.
    Sometimes it is the case that there is a bug unrelated to your changes that cause the tests to fail but we are aware of these.
    If you see one of these failures, feel free to ask!

*   Meanwhile, we'll also review your code.
    Some common review points are:
    *   This seems odd, why was this done?
    *   Could you see if you can use the functionality from place X?
    *   Could you create a test or documentation for this?
    *   This could be a nicer way of doing this, what do you think?
    Occasionally there will be some major point which will require more discussion but those are more on a case-by-case basis.

*   This process of review, fix and testing may go on a few times.
    The simplest way to reduce the time needed and to help us too is to run the tests, code formatting check and doc building locally.
    If they all pass locally they will very often have no issues in the automated tests.

    Once everyone is happy, it's time for us to hit (*squash*) merge, get it into the development branch and have one more contribution to auto-sklearn!

# Pull Request Overview
* Create a [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of the [automl/auto-sklearn](https://github.com/automl/auto-sklearn) git repo
* Check out what's available by running `make help`.
* Clone your own fork and create a new branch from the branch to work on
    ```bash
    git clone git@github.com:your-username/auto-sklearn.git
    cd auto-sklearn
    git checkout -b my_new_branch development

    # Initialize autosklearn/automl_common submodule
    git submodule update --init --recursive

    # Create a virtual environment and activate it so there are no package
    # conflicts
    python -m venv my-virtual-env
    source my-virtual-env/bin/activate

    make install-dev
    # pip install -e ".[test,docs,examples]" # To manually install things

    # Edit files...

    # Format code
    make format

    # Check for any issues
    make check

    # ... fix any issues

    # If you changed documentation:
    # This will generate all documentation and check links
    make doc
    make linkcheck
    make examples  # mainly needed if you modified some examples

    # ... fix any issues

    # If you edited any code
    # Check out pytest --help if you want to only run specific tests
    pytest

    # ... fix any issues

    # If you want to run pre-commit, the formatting checks we run on github
    pre-commit install
    pre-commit run --all-files

    # ... fix any issues

    # Check the changed files
    git status

    # Add the changes
    git add {changed files}
    git commit -m "Meaningful as you can make it message"

    # Push back to your fork
    git push --set-upstream origin my_new_branch
    ```
* Go to github, go to your fork and then make a pull request using the **Contribute** button.
    * `automl/auto-sklearn` | `development` <- `your-username/auto-sklearn` | `my_new_branch`
* Write a [PR](#creating-the-pr) with a description of the changes, why you implemented them and any implications.
    * Check out this [blog post](https://hugooodias.medium.com/the-anatomy-of-a-perfect-pull-request-567382bb6067) for some inspiration!
* Once we see this, we will run some automated tests on the pull request. These
tests are the same as the ones you can run manually and are mentioned in the
[test](#testing) section.
* We'll review the code and perhaps ask for some changes
* Once we're happy with the result, we'll merge it in!

# General FAQ

### I've finished my pull request or made new changes and want it reviewed, now what?
*   We'll actively monitor what pull requests are already in progress.
    Once you believe you pull request to be ready or you need some feedback, feel free to comment on your pull request, tagging @eddiebergman or @mfeurer and we'll provide feedback as soon as we can!

### I've been asked to rebase my pull request, why and what do I do?
*   It can often be the case that while you were working on some changes, we may have merged something new into the development branch.

    This can be a problem because the branch you created was based off the old development branch.
    This means there are new changes in the automl/auto-sklearn code base you don't have in your forked repo or locally.
    This is not always an issue, generally if different parts of the code were touched they can be merged safely.
    Either way, if we ask you to [rebase](https://www.atlassian.com/git/tutorials/merging-vs-rebasing), it is because it won't merge or we think there may have been overlapping changes between what you were working on and the new code put into the development branch.
*   First, update the development branch on **your fork**
    *   The easiest way to do this is go to *https://github.com/your-username/auto-sklearn*, navigate to the development branch and hit **fetch upstream**.
    *   This will make it so your fork of auto-sklearn is now up to date
*   Second, we need to go to our clone and pull in these new changes to development
    ```bash
    git checkout development
    git pull
    ```
* Lastly, we need to rebase the my_new_branch on top of the new development
    ```bash
    git checkout my_new_branch
    git rebase development
    ```
*   Now if there were no conflicts, that's it, you can continue as normal.
    If there are conflicts, you'll have to sort these out which you can find out how to do [here](https://docs.github.com/en/get-started/using-git/resolving-merge-conflicts-after-a-git-rebase) and [here](https://docs.github.com/en/github/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line).

