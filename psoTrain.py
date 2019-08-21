from datetime import datetime
import copy
import sys
import sqlite3
import argparse
import os
import time
import shutil

import dask
from distributed import LocalCluster
from distributed import Client, as_completed

from psoParticle import Particle, unpickle_position_velocity, score_particle_position
from config import REPLICATIONS, NUM_PARTICLES, ITERATIONS, \
    ITERATION_COLUMN_NAME, SCORE_COLUMN_NAME, POSITION_COLUMN_NAME, VELOCITY_COLUMN_NAME, RESULTS_TABLE_NAME
from HelpFunctions import error_print


class Train:
    """
    Runs a particle swarm optimisation algorithm
    """

    def __init__(self, dask_client, num_workers):
        self.dask_client = dask_client
        self.num_workers = num_workers
        self.replications = REPLICATIONS

        # get a dict of parameter ranges: {par_name: (min, max), ...}
        self.parameter_ranges = {"X": range(-100, 100), "Y": range(-200, 200, 2)}

        self.particles = [Particle(i, self.parameter_ranges) for i in range(NUM_PARTICLES)]
        self.particle_epochs_completed = [0] * len(self.particles)
        self.particles_running = [False] * len(self.particles)

        self.global_best_score = sys.float_info.max
        # to start, use the randomly generated position of the first particle
        self.global_best_position = copy.deepcopy(self.particles[0].position)

        self.iterations = ITERATIONS

        # check how many iterations of each particle have been done
        for particle_num, particle in enumerate(self.particles):
            if os.path.exists(particle.results_db_path_name):
                do_copy = True
                # assume db file exists (initialised by Particle)
                with sqlite3.connect(particle.results_db_path_name) as particle_score_db:
                    particle_score_db.execute("PRAGMA journal_mode=TRUNCATE")
                    query = "select {}, {}, {}, {} from {}".format(ITERATION_COLUMN_NAME, SCORE_COLUMN_NAME,
                                                                   POSITION_COLUMN_NAME, VELOCITY_COLUMN_NAME,
                                                                   RESULTS_TABLE_NAME)
                    try:
                        for row in particle_score_db.execute(query):
                            particle.update_score_position_velocity(row[1], row[2], row[3], pickled=True)
                            self.particle_epochs_completed[particle_num] = \
                                max(row[0], self.particle_epochs_completed[particle_num])
                            self.update_global(row[1], unpickle_position_velocity(row[2]))
                    except sqlite3.OperationalError as e:
                        # table doesn't exist so we don't need to do anything
                        error_print("--\nTrain init, particle {}\n{}\n--".format(particle_num, e))
                        do_copy = False
                if do_copy:
                    # make a copy in the tmp directory
                    shutil.copy(particle.tmp_db_path_name, particle.results_db_path_name)

    def update_global(self, score, position):
        if score < self.global_best_score:
            self.global_best_position = position
            self.global_best_score = score

    def solve(self):
        # start off some particles
        futures = []
        for particle in self.particles[:max(int((self.num_workers + 1) / self.replications), 1)]:
            futures.append(self.create_parallel_particle_future(particle.name))
            time.sleep(60)

        completed = as_completed(futures, with_results=True)

        for batch in completed.batches():
            for future, (particle_num, score, position, velocity) in batch:
                self.particles[particle_num].update_score_position_velocity(score, position, velocity)

                # particle not running anymore
                self.particles_running[particle_num] = False
                # update the epoch
                self.particle_epochs_completed[particle_num] += 1
                # see if there's a new best score
                self.update_global(score, position)

                print("-- tassie devil\n{}\n{}\n{}\n".format([particle.name for particle in self.particles],
                                                             self.particle_epochs_completed,
                                                             self.particles_running))

                # find the next particle (min epochs done)
                min_epochs = min(self.particle_epochs_completed)
                # print("min: {}".format(min_epochs))

                if min_epochs < self.iterations:
                    # not done yet - find the min particle
                    particle_num = self.particle_epochs_completed.index(min_epochs)
                    particle = self.particles[particle_num]

                    # print("index={}, particle_num={}".format(particle_num, particle.name))

                    # update for the next run
                    particle.update_velocity(self.global_best_position)
                    particle.update_position()

                    # score the particle position
                    completed.add(self.create_parallel_particle_future(particle_num))
        # do something with the results now
        print("=========== Done ({}) ===============".format(self.iterations))
        print(self)

    def create_parallel_particle_future(self, particle_num):
        self.particles_running[particle_num] = True
        # print("creating future for {}\n{}\n{}\n{}\n".format(particle_num,
        #                                                     [particle.name for particle in self.particles],
        #                                                     self.particle_epochs_completed,
        #                                                     self.particles_running))

        particle_epoch = self.particle_epochs_completed[particle_num]
        print("iteration {}: particle {} (score={})".format(particle_epoch, particle_num,
                                                            self.particles[particle_num].local_best_score))

        future = self.dask_client.submit(score_particle_position, self.particles[particle_num], particle_epoch)

        return future

    def __str__(self):
        train_string = "\n---{}---\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        train_string += "Best score: {}\n".format(self.global_best_score)
        train_string += "Best position: {}\n".format(self.global_best_position)
        for i, particle in enumerate(self.particles):
            train_string += "> {}: {}\n".format(i, round(particle.local_best_score, 3))
        return train_string


def setup_dask_client(num_cpus, arg_memory):
    # make the memory cutoffs more forgiving --- NOT WORKING
    with dask.config.set({"distributed.worker.memory.pause": False,
                          "distributed.worker.memory.spill": False,
                          "distributed.worker.memory.target": False,
                          "distributed.worker.memory.terminate": .999}):

        total_memory = arg_memory * 10 ** 9
        # work out resources for the local cluster
        mem_limit = round(total_memory / num_cpus)

        cluster = LocalCluster(n_workers=num_cpus, threads_per_worker=1, memory_limit=mem_limit)
        print([worker.memory_limit for worker in cluster.workers])

        dask_client = Client(cluster, timeout=600)

    return dask_client, cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Training Experiment.')
    parser.add_argument('-c', '--cpus', type=int, default=2,
                        help='CPUs to use (default: 2)')
    parser.add_argument('-m', '--memory', type=int, default=10,
                        help='Total memory available to the experiment in GB (default is 10)')

    args = parser.parse_args()
    print(args)

    client, cluster = setup_dask_client(args.cpus, args.memory)

    t = Train(client, args.cpus)
    t.solve()

    print("finished solving. closing client")

    if client is not None:
        client.close()
        cluster.close()
