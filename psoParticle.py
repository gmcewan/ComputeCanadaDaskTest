from bisect import bisect_left
import random
import sys
import pickle
import sqlite3
import math
from collections import defaultdict
import copy
import shutil
import os

from Experiment import Experiment
from config import SLURM_TMPDIR_STRING, REPLICATIONS, RESULTS_DB_PATH, RESULTS_TABLE_NAME, \
    RESULTS_DB_NAME, ITERATION_COLUMN_NAME, VELOCITY_COLUMN_NAME, PARTICLE_COLUMN_NAME, POSITION_COLUMN_NAME, \
    SCORE_COLUMN_NAME, OUTPUT_DIR, COGNITIVE, SOCIAL, CONSTRICTION, NUM_SCENARIOS


def take_closest(my_list, my_number):
    """
    Assumes my_list is sorted. Returns closest value to my_number.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(my_list, my_number)
    if pos == 0:
        value = my_list[0]
    elif pos == len(my_list):
        value = my_list[-1]
    else:
        before = my_list[pos - 1]
        after = my_list[pos]
        if after - my_number < my_number - before:
            value = after
        else:
            value = before
    return value


def score_particle_position(particle, epoch):
    score, position, velocity = particle.score_current_position(epoch)
    return particle.name, score, position, velocity


def pickle_position_velocity(dictionary):
    """
    Pickles a position or velocity dictionary  in a predictable order so it can be compared in
    the DB query.
    :param dictionary: A position or velocity dictionary {scope: {name: value}}
    :return: Pickle of the dictionary as a sorted list of tuples [(scope, name, value)]
    """
    tuple_list = []
    for name, value in dictionary.items():
        tuple_list.append((name, value))
    tuple_list.sort()
    return pickle.dumps(tuple_list)


def unpickle_position_velocity(pickle_tuple_list):
    dictionary = defaultdict(dict)
    tuple_list = pickle.loads(pickle_tuple_list)
    # print(tuple_list)

    for name, value in tuple_list:
        dictionary[name] = value

    return dictionary


class Particle:
    """
    Particle in a particle swarm optimisation. Has position and velocity. Can work out the error associated with
    a location.
    """

    def __init__(self, particle_name, parameter_ranges):
        self.name = particle_name
        self.parameter_ranges = parameter_ranges

        if SLURM_TMPDIR_STRING[1:] in os.environ:
            self.tmp_dir = os.environ[SLURM_TMPDIR_STRING[1:]]
        elif os.path.exists(os.path.join(os.environ["HOME"], "scratch")):
            self.tmp_dir = os.path.join(os.environ["HOME"], "scratch")
        else:
            self.tmp_dir = OUTPUT_DIR

        self.tmp_db_path_name = os.path.join(self.tmp_dir, RESULTS_DB_NAME.format(self.name))
        self.results_db_path_name = RESULTS_DB_PATH.format(self.name)

        # set up the results db if it doesn't exist
        with sqlite3.connect(self.tmp_db_path_name, timeout=60) as db:
            db.execute("PRAGMA journal_mode=TRUNCATE")
            create_table_str = """create table if not exists {} 
                                  ({}, {}, {} PRIMARY KEY ON CONFLICT REPLACE, {}, {})"""
            create_table_str = create_table_str.format(RESULTS_TABLE_NAME, ITERATION_COLUMN_NAME, PARTICLE_COLUMN_NAME,
                                                       POSITION_COLUMN_NAME, VELOCITY_COLUMN_NAME, SCORE_COLUMN_NAME)
            db.execute(create_table_str)
            # print("created results table: {}".format(create_table_str))

        if not os.path.exists(self.results_db_path_name):
            # make a copy in the results directory
            shutil.copy(self.tmp_db_path_name, self.results_db_path_name)

        # initialise
        self.position_scored = False  # True if the current position has been scored
        self.position, self.velocity = self.random_position_velocity()

        self.local_best_position = copy.deepcopy(self.position)
        self.local_best_score = sys.float_info.max

    def update_score_position_velocity(self, score, position, velocity, pickled=False):
        if score < self.local_best_score:
            self.local_best_score = score

            if pickled:
                position = unpickle_position_velocity(position)
                velocity = unpickle_position_velocity(velocity)

            self.local_best_position = position
            self.velocity = velocity

    def random_position_velocity(self):
        """
        Generate a random position within the parameter boundaries.
        :return: Position inside parameter ranges
        """
        position = {}
        velocity = {}
        for field, value_list in self.parameter_ranges.items():
            low = min(value_list)
            high = max(value_list)
            position[field] = round(random.uniform(low, high), 3)
            velocity[field] = 0
        return position, velocity

    def update_velocity(self, global_best_position):
        """
        Update velocity using local and global best so far scores
        :param global_best_position: the global best so far position
        """
        new_velocity = {}

        for name, old_v in self.velocity.items():
            r1 = random.random()
            r2 = random.random()
            new_velocity[name] = CONSTRICTION * (old_v
                                                 + COGNITIVE * r1
                                                 * (self.local_best_position[name]
                                                    - self.position[name])
                                                 + SOCIAL * r2
                                                 * (global_best_position[name]
                                                    - self.position[name]))
        self.velocity = new_velocity

    def update_position(self):
        """
        Update position according to the current velocity
        Truncated at parameter range boundaries.
        """
        for name, old_pos in self.position.items():
            new_val = old_pos + self.velocity[name]

            value_list = self.parameter_ranges[name]
            low, high = value_list[0], value_list[-1]

            if new_val < low:
                self.position[name] = low
                self.velocity[name] = 0
            elif new_val > high:
                self.position[name] = high
                self.velocity[name] = 0
            else:
                new_val = take_closest(value_list, new_val)
                self.position[name] = new_val
        self.position_scored = False

    def score_current_position(self, current_iteration):
        """
        Works out the error score for the current position. Sets local_best_score if it's better than the previous best.
        """
        if not self.position_scored:
            pickle_position = pickle_position_velocity(self.position)
            pickle_velocity = pickle_position_velocity(self.velocity)

            # check if we've already scored this position (unlikely but scoring is expensive)
            with sqlite3.connect(self.tmp_db_path_name, timeout=60) as score_db:
                search_command = "select {} from {} where {}=?".format(SCORE_COLUMN_NAME,
                                                                       RESULTS_TABLE_NAME,
                                                                       POSITION_COLUMN_NAME)
                results = score_db.execute(search_command, (pickle_position,)).fetchall()

            if len(results) == 0:
                # no previous score for this position - have to work it out

                # create the experiment
                experiment = Experiment("p_{}-e_{}".format(self.name, current_iteration), REPLICATIONS, self.tmp_dir)

                # add a bunch of scenarios (only the scenario count has any impact
                for i in range(NUM_SCENARIOS):
                    experiment.add_scenario("s_{}".format(i), {"foo": "bar", "baz": 4})

                # run the experiment
                result_db_name = experiment.run_experiment()
                print("scoring against: {}".format(result_db_name))

                # the experiment results are not important for the score in the test
                score = self.get_ackley_score()
            else:
                # there is a previous score for this position - use that instead
                score = results[0][0]

            # record the score in the database
            with sqlite3.connect(self.tmp_db_path_name, timeout=60) as score_db:
                insert_command = "insert or replace into {} ({}, {}, {}, {}, {}) values (?, ?, ?, ?, ?)".\
                    format(RESULTS_TABLE_NAME, ITERATION_COLUMN_NAME, PARTICLE_COLUMN_NAME, POSITION_COLUMN_NAME,
                           VELOCITY_COLUMN_NAME, SCORE_COLUMN_NAME)
                score_db.execute(insert_command, (current_iteration, self.name, pickle_position, pickle_velocity,
                                                  score))

            # flag that we now have a score for this position
            self.position_scored = True

            if score < self.local_best_score:
                self.local_best_score = score
                self.local_best_position = copy.deepcopy(self.position)

            # make a copy in the results directory
            shutil.copy(self.tmp_db_path_name, self.results_db_path_name)

        return self.local_best_score, copy.deepcopy(self.position), copy.deepcopy(self.velocity)

    def get_ackley_score(self):
        """
        Scores the current position using the Ackley function
        :return: Score of position values X and Y in the Ackley function.
        """
        x = self.position["X"]
        y = self.position["Y"]
        score = -20 * math.exp(-.2 * math.sqrt(.5 * (x ** 2 + y ** 2))) \
                - math.exp(.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + math.e + 20
        return score

    def __str__(self):
        return "best={}, {}".format(self.local_best_score, self.local_best_position)

