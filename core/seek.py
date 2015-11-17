import sys
import os
from subprocess import call


num_arguments = len(sys.argv)
if num_arguments == 3:
  # hello
  x = d
elif num_arguments == 2:
    # Extract files
  if os.path.isdir(sys.argv[1]) or os.path.isfile(sys.argv[1]):
    call(["./executor.rb", sys.argv[1]])
  else:
    print("work on bad parameter message")
