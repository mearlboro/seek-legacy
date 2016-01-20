---------------
Seek
==============

Topic modelling tool. Target userbase: The whole world

# Usage
# Python package
### Installation
---------------
Seek makes use of extensive libraries and external tools so as a result, we are 
providing you with a one time installer for Linux and OS X. 

1. Simply run `./install.sh`
and everything should be setup.

2. Afterwards, run the command `source activate.sh` to export all the required paths
and create all the required NLTK and Stanford data. 

3. Once you have downloaded all of the required data, you can comment any 
installation or download commands in activate.sh, keeping only the exported paths.

4. Navigate to `$NLTK_DATA/nltk_trainer` and run the following command: 
`python train_chunker.py treebank_chunk --classifier=NaiveBayes
5. Copy `seek5.ser.gz` to `$STANFORD_MODELS`
6. Almost there! Navigate to core and run python init.py. That will exported
all the necessary objects for your application.



### Basic capabilities
---------------
- Named Entity Recognition
- Topic Modelling
- Summarization
- Relationship extraction
### Basic usage

The executor is the main program if you will. However, not all of the capabilities
have been currently implemented in it. To bypass that, simply run

`python executor.py/linguist.py [flag] [path] {Optional arguments}`

---------------
### Further development
All of the capabilities will later be added to the executor for simplicity.

### Web app
The Web App provides much of the same capabilities via natural language interpretation.

# Current tasks
## Phase 1
- [x] PDF to plain text
- [ ] Handwritten stuff to plain text
- [ ] Text from images
- [X] Web scraping - Wikipedia(big no no), Arxiv(sensitive bastards), Imperial's Spiral repository(say what?)
- [X] Parsing plain text to LDA rules

***

## Phase 2
- [X] Gatherer - script that runs the integration algorithms and parses the plain text
- [ ] Analyser - takes plain text ^ connects to database
- [X] Assistant - Interface for document/query input, RL (terminal only)

***

## Phase 3
- [X] LDA
- [ ] Topic linking to database 
- [ ] Weights and links for semantic analysis, knowledge graph
### Extensions
- [ ] Grammar skills for consistent discourse
- [ ] Author and agent analysis
- [ ] Associating images with context/concepts

***

## Phase 4
- [X] Visual interface
- [X] Speech recognition

***

## Phase 5
- [ ] Feed
- [ ] Model for agent/person
- [ ] Moral model

***