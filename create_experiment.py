__author__ = 'Richard Diehl Martinez'
'''
Generates a new experiment; automatically creates a jupyter notebook with
prefilled section specifying a list of hyperparameters. If experiment already
exists then runs jupyter notebook.
'''

import click
import os, sys
from shutil import copyfile

@click.command()
@click.option('--experiment_type', type=click.Path())
@click.option('--experiment_name', type=click.Path())
def main(experiment_type, experiment_name):
	''' Creates experiment folder and copies over most recent json params.'''
	type_path = os.path.join(os.getcwd(), "experiments", experiment_type)
	if os.path.isdir(type_path):
		experiment_path = os.path.join(type_path, experiment_name)
		if os.path.isdir(experiment_path):
			raise Exception("{} already exists.".format(experiment_name))

		all_subdirs = [os.path.join(type_path,d) for d in os.listdir(type_path) if \
							os.path.isdir(os.path.join(type_path,d)) and '__' not in d]
		os.mkdir(experiment_path)
		if all_subdirs:
			latest_subdir = max(all_subdirs, key=os.path.getmtime)
			params_fp = os.path.join(latest_subdir, "experiment_params.json")
			notebook_fp = os.path.join(latest_subdir, "experiment.ipynb")
			copyfile(params_fp, os.path.join(experiment_path, "experiment_params.json"))
			copyfile(notebook_fp, os.path.join(experiment_path, "experiment.ipynb"))
		else:
			_ = open(os.path.join(type_path, "experiment_params.json"), 'w').close()
			_ = open(os.path.join(type_path, "experiment.ipynb"), 'w').close()
	else:
		os.mkdir(type_path)

		_ = open(os.path.join(type_path, "prepare_model.py"), 'w').close()
		_ = open(os.path.join(type_path, "__init__.py"), 'w').close()


if __name__ == '__main__':
	main()
