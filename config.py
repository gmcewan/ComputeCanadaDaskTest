import os
import math


SLURM_TMPDIR_STRING = "$SLURM_TMPDIR"

INPUT_DIR = "input"
OUTPUT_DIR = "output"

# parameters for our testing
ITERATIONS = 100
REPLICATIONS = 20
NUM_PARTICLES = 30
NUM_SCENARIOS = 20

RESULTS_DB_NAME = "particle_{}-training_scores.db"
RESULTS_DB_PATH = os.path.join(OUTPUT_DIR, RESULTS_DB_NAME)
RESULTS_TABLE_NAME = "Results"
ITERATION_COLUMN_NAME = "iteration"
PARTICLE_COLUMN_NAME = "particle"
POSITION_COLUMN_NAME = "position"
VELOCITY_COLUMN_NAME = "velocity"
SCORE_COLUMN_NAME = "score"

# PSO parameters
INERTIA = .8
COGNITIVE = 2.8
SOCIAL = 1.3
THETA = COGNITIVE + SOCIAL
CONSTRICTION = 2 / abs(2 - THETA - math.sqrt(THETA**2 - 4 * THETA))

