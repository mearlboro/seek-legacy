import os
import sys
import textract

# Use script by calling $ python extractor.py <source> <destination>
# By default destination folder is set to /deep-shit/Raw Text

if len(sys.argv) == 1 :
    print("extractor expects the following command \n extractor.py <src> \n extractor.py <src> <dest>")
    sys.exit(0)
dst = os.getcwd() + '/txt'
if len(sys.argv) > 1 :
    src = sys.argv[1]
    if len(sys.argv) > 2 :
        dst = sys.argv[2]

filepath, file_ext = os.path.splitext(src)
filename = os.path.basename(filepath)
if os.path.isfile(dst + '/' + filename + '.txt'):
    print("File " + filename + " already exists")
else:
    print("Converting " + filename)
    try:
        text = textract.process(src)
    except TypeError:
        next
    f = open(dst + '/' + filename + '.txt', 'w+')

    f.write(text.decode())

    f.close()
