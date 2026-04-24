'''
A command library help user upload their results to dashboard.

The scripts is now disabled because of no maintenance.
'''
#!/usr/bin/env python
import os
import os.path
import json
import argparse
import re

import requests
from . import _utils
from . import cli_constant as cli

DASHBOARD_URL = cli.DASHBOARD_URL
QUERY_URL = DASHBOARD_URL + "/get?id=%d"

def get_result_from_id(query_id):
	'''Query uploaded info from id'''
	pass

def clone_codes_from_commit(git_user, git_repo, git_commit):
	'''Download codes from commit'''
	pass

def download(args):
	'''Entrance of download'''
	pass

	# run model
	# result_path = "{}/result.json".format(code_dir)
	# old_time_stamp = 0
	# if os.path.exists(result_path):
	# 	old_time_stamp = os.path.getmtime(result_path)

	# os.system("bash {}/run_model.sh".format(extract_dir))
	# if not os.path.exists(result_path) or \
	# 	os.path.getmtime('{}/result.json'.format(code_dir)) <= old_time_stamp:
	# 	raise FileNotFoundError("New result file not found.")
	# print(json.load(open("{}/result.json".format(code_dir), "r")))
