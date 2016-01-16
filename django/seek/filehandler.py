import subprocess
import os
import nltk
from .questions import get_question_id, get_answer
from .commandhandler import handle_command
import logging

CORE_DIR = '/develop/Seek/core/'
WEB_DIR = '/develop/seekweb/'
logger = logging.getLogger('handler')
def extract(src):
        os.chdir(CORE_DIR)
        ret_code = subprocess.call(["python",
                                    "executor.py",
                                    "-l",
                                    src,
                                    "../../seekweb/files/",
                                    "extract"])
        os.chdir(WEB_DIR)
        if ret_code == 0:
                confirmation = "Thank you. I am reading your document now..."
        else:
                confirmation = "Something happened. Can you upload again?"
        return ("files/" + os.path.basename(src).split(".")[0] + ".txt", confirmation)

def handle_file(filename, name):
        logger.error('HANDLING FILE')
        path = "files/" + ''.join(name.split(' '))
        with open(path, 'wb+') as destination:
                for chunk in filename.chunks():
                        destination.write(chunk)
        converted = extract('../../seekweb/' + path)
        logger.error(converted)
        logger.error(os.getcwd())
        if path.split(".")[-1] != 'txt':
            os.system("rm -f " + path)

        return converted

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


cmd_dict = { 1: ('mostfreq' , ' '   ),
             2: ('topics'   , '0 10'),
             4: ('entities' , ' '   ),
             5: ('entities' , ' '   ),
             6: ('entities' , ' '   )}

def handle_command_old(command, session_info):
        # questions.py has a dictionary with questions linked to ids, we need to assign each id a call to the core
        if command == '':
                # upload a file
                print('bla')
        print(session_info)
        qid = get_question_id(command)

        # find if there is a file in the session info
        index = 1
        l = len(session_info)
        filename = session_info[-index]
        while not (os.path.isfile(filename) or index >= l):
                filename = session_info[-index]
                index += 1

        if os.path.isfile(filename):
                cmd  = cmd_dict[qid][0]
                args = cmd_dict[qid][1]

                p = subprocess.Popen(['python3', '../Seek/core/linguist.py', cmd, filename, args], stdout=subprocess.PIPE)

                return p.stdout.read()
        else:
                return "I'm sorry, you gave me nothing to read from."


def handle_question(question, session_info):
        return get_answer(question)
#        os.chdir(CORE_DIR)
#        # Looks for an answer from the analyser
#        ret_code = subprocess.call(["python",
#                                    "analyser.py",
#                                    "-a",
#                                    question])
#        return ret_code


