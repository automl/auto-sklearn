import os
import platform
import pwd
import subprocess
import shlex

import sklearn
import numpy
import scipy

from AutoML2015.data.data_io import vprint


def check_system_info():
    verbose = True
    # Method is called before we start our script
    # Get some system information
    try:
        architecture = platform.architecture()
        pltfrm = platform.platform()
        java_ver = platform.java_ver()
        linux = platform.linux_distribution()
        env = os.environ
        vprint(verbose, "Architecture: %s" % str(architecture))
        vprint(verbose, "Platform: %s" % str(pltfrm))
        vprint(verbose, "Java_ver: %s" % str(java_ver))
        vprint(verbose, "Linux: %s" % str(linux))
        vprint(verbose, "Environment: %s" % str(env))
        vprint(verbose, "sklearn %s" % str(sklearn.__version__))
        vprint(verbose, "numpy: %s" % str(numpy.__version__))
        vprint(verbose, "scipy: %s" % str(scipy.__version__))
        vprint(verbose, "Username: %s" % str(pwd.getpwuid(os.getuid()).pw_name))
    except:
        vprint(verbose, "Could not find platform information")
        pass

    try:
        call = "java -version"
        call = shlex.split(call)
        out = subprocess.check_output(call)
        vprint(verbose, "Java -version: %s" % str(out))
        call = "which java"
        call = shlex.split(call)
        out = subprocess.check_output(call)
        vprint(verbose, "which java: %s" % str(out))
    except Exception as e:
        vprint(verbose, "Error while calling 'java -version': %s" % str(e))
        vprint(verbose, "Could not find java information")
        pass

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ls_dir = root_dir
    try:
        call = "ls -la %s" % ls_dir
        call = shlex.split(call)
        out = subprocess.check_output(call)
        vprint(verbose, "ls -la here: %s" % str(out))
    except Exception as e:
        vprint(verbose, "Calling: ls -la %s" % ls_dir)
        vprint(verbose, "Error while calling 'ls -la here': %s" % str(e))
        pass

    ls_dir = os.path.join(root_dir, "lib")
    try:
        call = "ls -la %s" % ls_dir
        call = shlex.split(call)
        out = subprocess.check_output(call)
        vprint(verbose, "ls -la in /lib/: %s" % str(out))
    except Exception as e:
        vprint(verbose, "Calling: ls -la %s" % ls_dir)
        vprint(verbose, "Error while calling 'ls-la in lib': %s" % str(e))
        pass

    ls_dir = os.path.join(root_dir, "lib", "jre1.8.0_25", "bin")
    try:
        call = "ls -la %s" % ls_dir
        call = shlex.split(call)
        out = subprocess.check_output(call)
        vprint(verbose, "ls -la in /lib/jre/bin: %s" % str(out))
    except Exception as e:
        vprint(verbose, "Calling: ls -la %s" % ls_dir)
        vprint(verbose, "Error while calling 'ls-la in lib/jre/bin': %s" % str(e))
        pass