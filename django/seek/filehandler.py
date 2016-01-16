import subprocess
import os
import nltk
from .questions import get_question_id, get_answer
from .commandhandler import handle_command
import logging

CORE_DIR = '/develop/Seek/core/'
WEB_DIR = '/develop/Seek/django/'
FILES_DIR = '/develop/Seek/txt/'
logger = logging.getLogger('handler')


# -----------------------------------------------------------------------------
# Handles an user upload by uploading the file to the server than calling the executor to extract it

def handle_file(f, name):
        logger.error('HANDLING FILE')

	# uploads the file to the server than extracts it
        path = FILES_DIR + ''.join(name.split(' '))
        with open(path, 'wb+') as destination:
                for chunk in f.chunks():
                        destination.write(chunk)
        converted = extract(path)

        logger.error(converted)
	# clears all raw files, keeps the txt
        if path.split(".")[-1] != 'txt':
            os.system("rm -f " + path)

        return converted

def extract(src):
        os.chdir(CORE_DIR)
        ret_code = subprocess.call(["python",
                                    "executor.py",
                                    "-l",
                                    src,
                                    FILES_DIR,
                                    "extract"])
        os.chdir(WEB_DIR)
        if ret_code == 0:
                confirmation = "Thank you. I am reading your document now..."
        else:
                confirmation = "Something happened. Can you upload again?"
        return (FILES_DIR + os.path.basename(src).split(".")[0] + ".txt", confirmation)



# -----------------------------------------------------------------------------
# Handles any user verbal or textual input

def handle_user_input(query, file_list, user_name):
    (ret, answer, name) = handle_command(query, file_list, user_name)
    logger.error(ret)
    if ret == 0:
        return (ret, answer, name)
    if ret == 2:
        return (ret, "Hello " + name, name)
    answer = get_answer(query)
    logger.error(answer)
    return (0, answer, name)



