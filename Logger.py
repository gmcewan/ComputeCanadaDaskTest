import sqlite3
from collections import defaultdict
import os.path
import datetime

from HelpFunctions import error_print


class Logger:
    """
    Logs data to the database.
    """

    @staticmethod
    def local_db_name(experiment_name, scenario_name, run_id):
        names = [str(x).strip().replace(" ", "_") for x in [experiment_name, scenario_name, run_id]]
        name = "{}-sc_{}-rep_{}.db".format(*names)
        return name

    def __init__(self, experiment_name, scenario_name, run_id, scratch_path):
        """
        Start up a logger to record information for a unique model run.
        The DB unique name is made up of the experiment name, scenario name, and run ID.

        :param experiment_name: Name of this experiment. Used to name the database.
        :param scenario_name: Name of the scenario. Used to name the database.
        :param run_id: ID of this replication/run. Used to name the database.
        :param scratch_path: Relative or absolute path for location to create DB.
        """
        self.experiment_name = experiment_name
        self.scenario_name = scenario_name
        self.run_id = run_id

        self.end_step = 0

        self.db_name = self.local_db_name(experiment_name, scenario_name, run_id)

        self.db_dir = os.path.abspath(scratch_path)
        self.db_filepath = os.path.join(self.db_dir, self.db_name)

        with sqlite3.connect(self.db_filepath) as log_db:
            log_db.execute("PRAGMA journal_mode=OFF")

        if os.path.exists(self.db_filepath):
            os.remove(self.db_filepath)

        self.tables = {}  # {table_name: 1}

        self._property_loggers = defaultdict(set)

    def create_log_table(self, table_name):
        """
        Creates a table in the database for logging a type of information.

        :param table_name: Name of the table to create.
        """
        with sqlite3.connect(self.db_filepath, timeout=60) as log_db:
            create_str = """CREATE TABLE IF NOT EXISTS {} 
                                        (scenario_name, run_id, type, time, value,
                                        unique (scenario_name, run_id, type, time) 
                                        ON CONFLICT REPLACE)""" \
                                        .format(table_name)
            log_db.execute(create_str)
        self.tables[table_name] = 1

    def log_info(self, info_type, info_string, timestamp, value):
        """
        Logs a single bit of information from the base_model.

        :param info_type: String name for the type of information being logged (table name)
        :param info_string: String description of this entry
        :param timestamp: Time being logged (days)
        :param value: The information to be recorded.
        """
        if info_type not in self.tables:
            self.create_log_table(info_type)
        # try:
        insert_str = "INSERT INTO {} VALUES (?, ?, ?, ?, ?)".format(info_type)

        with sqlite3.connect(self.db_filepath, timeout=60) as log_db:
            try:
                log_db.execute(insert_str, (self.scenario_name, self.run_id, info_string, timestamp, value))
            except sqlite3.OperationalError as e:
                error_print("---")
                error_print("{}: failed to log to {}".format(datetime.datetime.now(), self.db_filepath))
                error_print(" Table={}: ({}, {}, {}, {}, {})".format(info_type,
                                                                     self.scenario_name,
                                                                     self.run_id,
                                                                     info_string,
                                                                     timestamp,
                                                                     value))
                error_print(e)

    def end_scenario(self, time_step):
        """
        Caps off all the tables with -1

        :param time_step: The time value for the capping entries
        """
        with sqlite3.connect(self.db_filepath, timeout=60) as log_db:
            for table_name in self.tables:
                insert_str = "INSERT INTO {} VALUES (?, ?, ?, ?, ?)".format(table_name)
                log_db.execute(insert_str, (self.scenario_name, self.run_id, "", time_step, -1))

    def close(self):
        """
        Close down the logger safely.
        """
        pass
        # self.db.commit()
        # self.db.close()

        # del self.db

    @staticmethod
    def gather_databases(experiment_name, loggers_info):
        """
        Collects entries from multiple databases from different models in an experiment.

        An experiment will run many models (for scenarios and replications), and each one will have it's
        own DB. A single experiment DB is required for analysis.

        :param experiment_name: Name of the experiment. Used as the name of the output DB.
        :param loggers_info: List specifying all the model DBs:
                             [(model_db_filepath, [table_name, table_name, ...]), ...]
        """
        print("-- gathering databases: {}".format(experiment_name))
        out_db_filename = os.path.join(os.path.dirname(loggers_info[0][0]), "{}.db".format(experiment_name))
        if os.path.exists(out_db_filename):
            os.remove(out_db_filename)
        # print("** {}".format(out_db_filename))
        with sqlite3.connect(out_db_filename, timeout=60) as out_db:
            out_db.execute("PRAGMA journal_mode=OFF")
            for in_db_filepath, tables_names in loggers_info:
                with sqlite3.connect(in_db_filepath, timeout=60) as in_db:
                    # go through all the tables
                    for info_type in tables_names:
                        # print("  {}".format(info_type))
                        # create the table if it's not there
                        create_str = """CREATE TABLE IF NOT EXISTS {} 
                                          (scenario_name, run_id, type, time, value,
                                          unique (scenario_name, run_id, type, time)
                                          ON CONFLICT REPLACE)""".format(info_type)
                        out_db.execute(create_str)

                        # all the rows
                        for row in in_db.execute("""select * from {}""".format(info_type)):
                            out_db.execute("insert into {} values (?, ?, ?, ?, ?)".format(info_type), (*row,))
                # remove the replication ("in") DB as we don't need it anymore
                try:
                    os.remove(in_db_filepath)
                    os.remove(in_db_filepath + "-journal")
                except FileNotFoundError:
                    # in DB and/or journal file doesn't exist - ignore and move on
                    pass
        return out_db_filename
