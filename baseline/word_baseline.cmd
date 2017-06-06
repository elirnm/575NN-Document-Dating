executable = /usr/bin/python3
getenv = true
output = word_baseline.results
error = condor.err
log = condor.log
arguments = word_baseline.py
transfer_executable = false
notification = complete
queue
