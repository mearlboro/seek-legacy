# This script requires the textract package,
# which you can download following http://textract.readthedocs.org/en/latest/installation.html
import os
import sys
import textract


if len(sys.argv) == 1 : sys.exit(0)
dst = os.getcwd()
print dst
if len(sys.argv) > 1 :
    src = sys.argv[1]
    if len(sys.argv) > 2 :
        dst = sys.argv[2]

filepath, file_ext = os.path.splitext('src')
filename = os.path.basename(filepath)

text = textract.process(src)
f = open(dst + '/' + filename + '.txt', 'w+')
f.write(text)
f.close()
