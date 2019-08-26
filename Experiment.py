from dask.distributed import get_client
from dask.distributed import secede, rejoin

import os.path
import shutil
import random

from Model import run_model
from Logger import Logger
from config import OUTPUT_DIR


class Experiment:
    """
    An experiment runs the base_model with varying parameters, with multiple replications of each parameter set.
    """

    def __init__(self, name, num_replications, scratch_path):
        self.scratch_path = scratch_path

        # get rid of any incidental spaces in the experiment name
        self.name = name.replace(" ", "_")

        self.num_replications = num_replications

        self.scenarios = []

        self.model_times = {}  # {(model_id, run_id): days, ...}

    def add_scenario(self, scenario_name, parameter_dict):
        """
        Adds a scenario for this experiment to run. The scenario contains information for running a model.

        :param scenario_name: string name of this scenario
        :param parameter_dict: dictionary of parameter files for the scenario
        """
        self.scenarios.append((scenario_name, parameter_dict))

    def run_experiment(self):
        """
        Run the experiment. Including all scenarios and replications.
        """
        dask_client = get_client(timeout=600)

        # print("dask client: {}".format(dask_client))
        # try:
        secede()
        print("secede: {}".format(self.name))
        #     seceded = True
        # except ValueError:
        #     seceded = False

        futures = []
        total_runs = len(self.scenarios) * self.num_replications
        for scenario in self.scenarios:
            scenario_name, configuration = scenario

            # replications_done = self.find_replications_done(scenario_name, configuration)
            replications_done = 0

            # random number of steps for this scenario
            total_steps = random.randint(300, 700)

            for run_id in range(replications_done, self.num_replications):
                future = dask_client.submit(run_model, self.name, scenario_name, run_id, total_runs, total_steps,
                                            self.scratch_path)
                futures.append(future)

        loggers_info = dask_client.gather(futures)

        # if seceded:
        print("rejoin: {}".format(self.name))
        rejoin()

        # gather all the output dbs into a single db
        out_db_filepath = Logger.gather_databases(self.name, loggers_info)

        dest_file_path_name = os.path.join(OUTPUT_DIR, os.path.split(out_db_filepath)[1])

        if not os.path.exists(dest_file_path_name):
            shutil.move(out_db_filepath, dest_file_path_name)

        return dest_file_path_name

    def add_model_time(self, result):
        self.model_times[result[:2]] = result[2]
        print(result)

    @staticmethod
    def callback_error(error):
        print(error)


