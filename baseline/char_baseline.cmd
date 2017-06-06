executable = /usr/bin/python3
getenv = true
output = char_baseline.results
error = condor.err
log = condor.log
arguments = char_baseline.py
transfer_executable = false
notification = complete
queue
