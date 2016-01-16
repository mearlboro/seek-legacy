import nltk
import os 
import subprocess
import string
import logging

CORE_DIR = '/develop/Seek/core/'
WEB_DIR = '/develop/seekweb/'
logger = logging.getLogger('handler')

# looks into the file stack and grabs the most recent one
# deepens recursively in case a file is not found
def grab_most_recent_file(file_list):
    index = 1
    l = len(file_list)
    if l > 0:
        filename = file_list[-index]
        while not (os.path.isfile(filename) or index >= l):
            filename = file_list[-index]
            index += 1
    else:
        filename = ''
    return filename

# --------------------------------------------------------------------------------
# calls the core with the command given in <cmd>
def call_core(cmd, args, filename):
    os.chdir(CORE_DIR)
    p = subprocess.Popen(['python3', 
                          'linguist.py', 
                          cmd, 
                          filename] + args, 
                          stdout=subprocess.PIPE)
    os.chdir(WEB_DIR)
    return p.stdout.read()


# --------------------------------------------------------------------------------
summary_words  = [ 'summarize', 'summarise', 'overview', 'short', 'summary', 'synopsis' ]
similar_words  = [ 'similar', 'related' ]
mostfreq_words = [ 'most', 'frequent', 'word' ]
people_words   = [ 'people', 'person', 'persons', 'who', 'names', 'individuals', 'humans']
places_words   = [ 'places', 'locations', 'location', 'place', 'area' ]
orgs_words     = [ 'organization', 'organizations' ]
topics_words   = [ 'about', 'subject', 'subjects', 'topic', 'topics' ] 
name_words     = [ 'my', 'name', 'is' ]
name_words_2   = [ 'i', 'am' ]

# finds what command to call, and then handles it
def handle_command(command, file_list, user_name):
    return_status = 0
    answer = ''
    name = user_name

    # filter through the user question to see what he wants
    command = ' '.join(command.split(string.punctuation))
    words = nltk.word_tokenize(command)

    is_name      = all(map(lambda w: w.lower in words, name_words))
    is_name_2    = all(map(lambda w: w.lower in words, name_words_2))

    req_summary  = any(map(lambda w: w.lower() in words, summary_words ))
    req_similar  = any(map(lambda w: w.lower() in words, similar_words ))
    req_mostfreq = all(map(lambda w: w.lower() in words, mostfreq_words))
    req_people   = any(map(lambda w: w.lower() in words, people_words))
    req_places   = any(map(lambda w: w.lower() in words, places_words))
    req_orgs     = any(map(lambda w: w.lower() in words, orgs_words  ))
    req_topics   = any(map(lambda w: w.lower() in words, topics_words))

    # if an introduction was found
    if(is_name):
        name = words[3]
        logger.error(name)
        return_status = 2
    elif(is_name_2):
        name = words[2]
        logger.error(name)
        return_status = 2
    # if any command was found
    elif any([req_summary, req_similar, req_mostfreq, req_people, req_places, req_topics, req_orgs]):
        # grab the file to process
        logger.error('FILE LIST')
        logger.error(file_list)
        filename = grab_most_recent_file(file_list)
        logger.error(filename)
        logger.error(os.getcwd())
        if os.path.isfile(filename):
            args = []
            filename = '../../seekweb/' + filename 
            if(req_summary): 
              cmd  = 'summary'
              args = ['0' , '10']
            if(req_similar):
              cmd = 'similar'
            if(req_mostfreq):
              cmd = 'mostfreq'
            if(req_people):
              cmd = 'entities'
              args = ['0', '0'] 
            if(req_places):
              cmd = 'entities'
              args = ['0', '1']
            if(req_orgs):
              cmd = 'entities'
              args = ['0', '3']
            if(req_topics):
              cmd = 'topics'
              args = ['1', '3']

            logger.error(cmd)
            if cmd != '':
                answer = call_core(cmd, args, filename) 
            else:
                return_status = -1
        else:
            answer = "I'm sorry, " + user_name + ". You gave me nothing to read from. Would you like me to seek somewhere else?"
    else:
        return_status = 1

    return (return_status, answer, name)
