"""Mock resource module for Windows"""
import sys

# Constants
RLIMIT_CPU = 0
RLIMIT_FSIZE = 1
RLIMIT_DATA = 2
RLIMIT_STACK = 3
RLIMIT_CORE = 4
RLIMIT_RSS = 5
RLIMIT_NPROC = 6
RLIMIT_NOFILE = 7
RLIMIT_OFILE = RLIMIT_NOFILE
RLIMIT_MEMLOCK = 8
RLIMIT_AS = 9
RLIMIT_LOCKS = 10
RLIMIT_SIGPENDING = 11
RLIMIT_MSGQUEUE = 12
RLIMIT_NICE = 13
RLIMIT_RTPRIO = 14
RLIMIT_RTTIME = 15

def getrlimit(resource):
    """Mock getrlimit"""
    return (1024, 1024)

def setrlimit(resource, limits):
    """Mock setrlimit"""
    pass

def getrusage(who):
    """Mock getrusage"""
    class RUsage:
        ru_utime = 0.0
        ru_stime = 0.0
        ru_maxrss = 0
        ru_ixrss = 0
        ru_idrss = 0
        ru_isrss = 0
        ru_minflt = 0
        ru_majflt = 0
        ru_nswap = 0
        ru_inblock = 0
        ru_oublock = 0
        ru_msgsnd = 0
        ru_msgrcv = 0
        ru_nsignals = 0
        ru_nvcsw = 0
        ru_nivcsw = 0
    return RUsage()

def prlimit(pid, resource, limits=None):
    """Mock prlimit"""
    if limits is None:
        return (1024, 1024)
    return None

# Create struct_rlimit if needed
class struct_rlimit(tuple):
    pass

# Patch into sys.modules
sys.modules['resource'] = sys.modules[__name__]
