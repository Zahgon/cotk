'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import subprocess
from subprocess import PIPE
import re
import sys

import requests

def assert_repo_exist():
	'''Assert cwd is in a git repo.'''
	pass

def check_repo_clean():
	'''Check whether repo is clean.
	Return True if clean, False if dirty.'''
	pass

def get_repo_workingdir():
	'''Get relative path of cwd from git repo root.'''
	pass

def get_repo_root_path():
	pass

def get_repo_remote():
	'''Get remote repo name on github'''
	pass

def get_repo_commit():
	'''Return the commit sha of HEAD'''
	pass

def assert_commit_exist(git_user, git_repo, git_commit):
	'''Assert commit is available'''
	pass

def git_clone(git_user, git_repo):
	pass

def git_checkout_commit(git_commit):
	pass
