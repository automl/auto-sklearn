import os
import pwd

import psutil

from data.data_io import vprint
verbose = True


def filter_pick(str_list, filter):
    # iterates over list and checks for each entry, whether filter matches
    match = list()
    try:
        match = [l for l in str_list for m in (filter(l),) if m]
    except IndexError:
        return False
    if len(match) == 1:
        return True
    else:
        return False


def send_signal_to_our_processes(filter, sig=0):
    # Sends sig to all processes matching filter
    processes = psutil.process_iter()
    for proc in processes:
        if proc.username() == pwd.getpwuid(os.getuid()).pw_name:
            # = It is our process
            if filter_pick(str_list=proc.cmdline(), filter=filter):
                vprint(verbose, "Sending %d to %s" % (sig, proc.cmdline()))
                proc.send_signal(sig)


def _change_permission(cmd):
    try:
        return_code = os.system(cmd)
    except:
        return_code = -1
    return return_code


def change_permission_folder(fl, perm="755"):
    if not os.path.isdir(fl):
        vprint(verbose, "%s is not a directory" % fl)
        return -1
    cmd = "chmod %s %s/* -R" % (perm, fl)
    return _change_permission(cmd)


def change_permission_file(fl, perm="755"):
    if not os.path.isfile(fl):
        vprint(verbose, "%s is not a file" % fl)
        return -1

    cmd = "chmod %s %s" % (perm, file)
    return _change_permission(cmd)
