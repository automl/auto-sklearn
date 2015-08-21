# -*- encoding: utf-8 -*-
import fnmatch
import os
import subprocess

from conf.settings import BINARIES_DIRECTORY


def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def find_files(folder, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def _clojure_search_file_in_folder(default_folder):
    folder = default_folder

    def search_file(prog_name):
        try:
            files = find_files(folder, prog_name)
            assert files
            result = os.path.normpath(files[0])
        except Exception as e:
            result = None
        return result

    return search_file


search_prog_in_binaries = _clojure_search_file_in_folder(BINARIES_DIRECTORY)


def search_prog(prog_name):
    try:
        p = subprocess.Popen(["whereis", prog_name], stdout=subprocess.PIPE)
        (output, _) = p.communicate()
        assert output, "Not found %s" % prog_name
        paths = output.split()
        assert paths, "Not found (whereis) prog - %s" % prog_name
        result = paths[1]
    except Exception as e:
        result = None
    return result
