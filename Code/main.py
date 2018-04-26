from agent import SharedAgent
from utils import *

import json
import pdb

def main(args):
	# Parse command-line arguments.
	args = parse_arguments()

	with open(args.config) as f:
		params = json.load(f)

	create_folder(params['trained_weights_path'])

	agent = SharedAgent(params)

	if eval(params['train']):
		agent.train()
	else:
		agent.test()

if __name__ == '__main__':
	main(sys.argv)
