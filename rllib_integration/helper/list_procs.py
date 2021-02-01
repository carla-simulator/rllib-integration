# """
# Usage:
#
# >>> for pid, name in search_procs_by_name("python").items():
# ...     print(pid, name)
# ...
# 11882 python3.6
# 47599 python3.6
# 51877 python3.6
# 51924 python3.6
#
# You can use regular expressions as well:
#
# >>> for pid, name in search_procs_by_name("p.t..n").items():
# ...     print(pid, name)
# ...
# 11882 python3.6
# 47599 python3.6
# 51877 python3.6
# 51924 python3.6
#
# You can search by command line arguments:
#
# >>> for pid, cmdline in search_procs_by_cmdline("tensorboard").items():
# ...     print(pid, cmdline)
# ...
# 51924 ['/Users/blabla/.virtualenvs/tf2/bin/python3.6', '/Users/blabla/.virtualenvs/tf2/bin/tensorboard', '--logdir=./my_logs', '--port=6006']
# """

import psutil


def proc_names():
    return dict([(proc.pid, proc.name()) for proc in psutil.process_iter()])


def proc_cmdlines():
    cmdlines = {}
    for proc in psutil.process_iter():
        try:
            cmdlines[proc.pid] = proc.cmdline()
        except psutil.AccessDenied:
            cmdlines[proc.pid] = None
    return cmdlines


def to_regex(regex):
    if not hasattr(regex, "search"):
        import re

        regex = re.compile(regex)
    return regex


def search_procs_by_name(regex):
    pid_names = {}
    regex = to_regex(regex)
    for pid, name in proc_names().items():
        if regex.search(name):
            pid_names[pid] = name
    return pid_names


def search_procs_by_cmdline(regex):
    pid_cmdlines = {}
    regex = to_regex(regex)
    for pid, cmdline in proc_cmdlines().items():
        if cmdline is not None:
            for part in cmdline:
                if regex.search(part):
                    pid_cmdlines[pid] = cmdline
                    break
    return pid_cmdlines
