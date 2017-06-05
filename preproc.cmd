executable = /usr/bin/python3
getenv = true
output = condor.out
error = condor.err
log = condor.log
arguments = preprocess.py
transfer_executable = false
notification = complete
queue
