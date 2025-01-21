#!/bin/bash 

#gitprompt.bash
#Modify prompt if we are in a git directory to show our branch
#2016-10-05
#Fergal Mullally

#When, in a terminal, you cd to a directory that is part of a git
#repo, this script changes your prompt to show that 
#a) You are in a git repo
#b) What branch you are one
#c) Your path relative to the root of the repo (not the filesystem)
#
#When you are not in a repo, it shows a default prompt (which you may
#want to change to your own tastes
#
#To use this script, put it in a directory on your $PATH, and add the 
#following command to your .bashrc
#PROMPT_COMMAND=source gitprompt.bash

#The default prompt for when you are not in a git repo
DEFAULT_PROMPT="Lios $CONDA_PROMPT_MODIFIER $VIRTUAL_ENV_PROMPT\W> "

bold=$(tput bold)
normal=$(tput sgr0)

branch=`git branch 2>/dev/null`
if [ -z "$branch" ]
then
	#Not in git
	PS1=$DEFAULT_PROMPT
else
	#In git
	currentBranch=`grep '\*' <<< "$branch" | awk '{print $NF}'`
	
	topLevelPath=`git rev-parse --show-toplevel`
	#relPath=`sed 's|'$topLevelPath'|.|' <<< $PWD`
	relPath=`awk -F '/' '{print $NF}' <<< $PWD`
	PS1="Lios $CONDA_PROMPT_MODIFIER $VIRTUAL_ENV_PROMPT $currentBranch::$relPath> "
fi

export PS1


	
