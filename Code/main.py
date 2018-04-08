from agent import Agent
from utils import *

import json

def main(args):
	# Parse command-line arguments.
	args = parse_arguments()

	with open(args.config) as f:
		params = json.load(f)
	
	create_folder(params['trained_weights_path'])

	agent = Agent(params)
	
	if eval(params['train']):
		agent.train()
	else:
		agent.test(params['test_data_path'])

if __name__ == '__main__':
	main(sys.argv)