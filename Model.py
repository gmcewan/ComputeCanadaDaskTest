from datetime import datetime
import itertools
import copy
import random
import time

from Logger import Logger


def run_model(experiment_name, scenario_name, run_id, total_runs, total_steps, scratch_path):
    # print("    scenario {}: rep {} started".format(scenario_name, run_id))

    model = Model(experiment_name, scenario_name, run_id, scratch_path)

    from timeit import default_timer as timer
    start = timer()

    logger_filepath, logger_tables = model.run(total_runs, total_steps)

    del model

    end = timer()
    minutes, seconds = divmod(end - start, 60)
    hours, minutes = divmod(minutes, 60)

    # print("      Scenario_{}-rep_{} took {}h:{:0>2}m:{:0>2}s".format(scenario_name, run_id, hours, minutes,
    #                                                                  round(seconds)))

    return logger_filepath, logger_tables


class Model:
    """
    A generic model. Manages generating data and recording for one scenario replication.
    """

    def __init__(self, experiment_name, scenario_name, run_id, scratch_path):
        self.id_counter = itertools.count()

        self.experiment_name = experiment_name
        self.scenario_name = scenario_name
        self.run_id = run_id

        # set up a logger to use - only need one per base_model
        self.logger = Logger(experiment_name, scenario_name, run_id, scratch_path)

        self.step_counter = itertools.count()
        self.current_step = next(self.step_counter)
        self._total_steps = None
        self._total_days = None

        # true if the model has started but not ended
        self.started = False

    @property
    def total_days(self):
        return self._total_days

    @total_days.setter
    def total_days(self, value):
        # only get bigger
        if self._total_days is None or value > self._total_days:
            self._total_days = value
            self._total_steps = value * 4

    @property
    def total_steps(self):
        return self._total_steps

    @total_steps.setter
    def total_steps(self, value):
        if self._total_steps is None or value > self._total_steps:
            self._total_steps = value
            self._total_days = round(value / 4)

    def get_new_id(self):
        """
        Get a number that is unique in this base_model.

        :return: New unique number
        """
        return next(self.id_counter)

    def run(self, total_replications, total_model_steps):
        """
        Run the base_model. Calls start(), then step() until done, then end()

        :param total_replications: Total number of replications of this scenario (just for friendly prints)
        :param total_model_steps: Total number of steps to run the model
        """
        self.total_steps = total_model_steps

        self.start(total_replications)
        self.run_to_step()
        if self.started:
            self.end()
        return copy.deepcopy(self.logger.db_filepath), copy.deepcopy(self.logger.tables)

    def start(self, total_runs):
        # print("start {}:{}:{}  total={} ({})".format(self.experiment_name, self.scenario_name, self.run_id, total_runs,
        #                                              datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.started = True

    def run_to_step(self, end_step=None):
        """
        Run the model for a set number of steps.

        :param end_step: Step to end on. Defaults to the total steps.
        """
        info_types = ["count", "resistance", "treatments"]

        if end_step is None:
            end_step = self.total_steps
        while self.current_step < end_step and self.started:
            # sleep for a random time 50-200 milliseconds
            sleep_time = random.uniform(.05, .2)
            time.sleep(sleep_time)

            # log something to the database
            self.logger.log_info(random.choice(info_types), "test", self.current_step, random.randint(10, 1000))

            # increment the step
            self.current_step = next(self.step_counter)

    def end(self):
        """
        Stop the model running. Closes everything safely.
        """
        # print("end {}:{}:{}  --  {}/{} steps".format(self.experiment_name, self.scenario_name, self.run_id,
        #                                              self.current_step, self.total_steps))
        # self.current_step = self.total_steps
        self.logger.end_scenario(self.current_step + 1)

        self.logger.close()

        # print("End: {} {} - {} ({})".format(self.experiment_name, self.scenario_name,
        #                                     self.run_id,
        #                                     datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.started = False
