import numpy as np
import mnist

from ml_genn import Network, Population, Connection
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EPropCompiler
from ml_genn.connectivity import Dense, ToroidalGaussian2D
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam

from time import perf_counter
from ml_genn.utils.data import (
    calc_latest_spike_time,
    calc_max_spikes,
    log_latency_encode_data
)
from ml_genn.compilers.eprop_compiler import default_params


# =========================
# Geometry
# =========================

INPUT_SHAPE = (28, 28, 1)
HIDDEN1_SHAPE = (28, 28, 1)
HIDDEN2_SHAPE = (16, 16, 1)
HIDDEN3_SHAPE = (8, 8, 1)

NUM_INPUT = np.prod(INPUT_SHAPE)
NUM_HIDDEN_1 = np.prod(HIDDEN1_SHAPE)
NUM_HIDDEN_2 = np.prod(HIDDEN2_SHAPE)
NUM_OUTPUT = 10


# =========================
# User-Controlled Parameters
# =========================

SIGMA_IN = 0.25
SIGMA_H = 0.25

DESIRED_FAN_IN_IN = int(NUM_INPUT*0.1)
DESIRED_FAN_IN_H1 = int(NUM_HIDDEN_1*0.1)
DESIRED_FAN_IN_H2 = int(NUM_HIDDEN_2*0.1)

# =========================
# Compute p_max analytically
# =========================

def compute_p_max(desired_fan_in, n_pre, sigma):
    return desired_fan_in / (n_pre * 2.0 * np.pi * sigma**2)


p_max_in = compute_p_max(DESIRED_FAN_IN_IN, NUM_INPUT, SIGMA_IN)
p_max_h1 = compute_p_max(DESIRED_FAN_IN_H1, NUM_HIDDEN_1, SIGMA_H)
p_max_h2 = compute_p_max(DESIRED_FAN_IN_H1, NUM_HIDDEN_1, SIGMA_H)

# Clip in case σ is very small
p_max_in = min(p_max_in, 1.0)
p_max_h1 = min(p_max_h1, 1.0)
p_max_h2 = min(p_max_h2, 1.0)


# =========================
# Weight scaling
# =========================

std_in = 1.0 / np.sqrt(DESIRED_FAN_IN_IN)
std_h1 = 1.0 / np.sqrt(DESIRED_FAN_IN_H1)
std_h2 = 1.0 / np.sqrt(DESIRED_FAN_IN_H2)


# =========================
# Training parameters
# =========================

BATCH_SIZE = 32
NUM_EPOCHS = 10
TRAIN = True

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels()
spikes = log_latency_encode_data(
    mnist.train_images(),
    20.0,
    51
)

serialiser = Numpy("latency_mnist_toroidal")

network = Network(default_params)

with network:

    input_pop = Population(
        SpikeInput(max_spikes=BATCH_SIZE * calc_max_spikes(spikes)),
        INPUT_SHAPE
    )

    hidden_1 = Population(
        LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0, tau_refrac=5.0),
        HIDDEN1_SHAPE
    )

    hidden_2 = Population(
        LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0, tau_refrac=5.0),
        HIDDEN2_SHAPE
    )

    hidden_3 = Population(
        LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0, tau_refrac=5.0),
        HIDDEN3_SHAPE
    )

    output = Population(
        LeakyIntegrate(tau_mem=20.0, readout="var"),
        NUM_OUTPUT
    )

    # -------------------------
    # Sparse toroidal connections
    # -------------------------

    Connection(
        input_pop,
        hidden_1,
        ToroidalGaussian2D(
            sigma=SIGMA_IN,
            p_max=p_max_in,
            weight=Normal(sd=std_in)
        )
    )

    Connection(
        hidden_1,
        hidden_2,
        ToroidalGaussian2D(
            sigma=SIGMA_H,
            p_max=p_max_h1,
            weight=Normal(sd=std_h1)
        )
    )

    Connection(
        hidden_2,
        hidden_3,
        ToroidalGaussian2D(
            sigma=SIGMA_H,
            p_max=p_max_h2,
            weight=Normal(sd=std_h2)
        )
    )

    Connection(
        hidden_3,
        output,
        Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN_2)))
    )

    # Random feedback for eProp
    Connection(
        hidden_1,
        output,
        Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
        feedback_name="f1"
    )

    Connection(
        hidden_2,
        output,
        Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
        feedback_name="f2"
    )    

    Connection(
        hidden_3,
        output,
        Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
        feedback_name="f3"
    )



# =========================
# Compile & Train
# =========================

max_example_timesteps = int(np.ceil(calc_latest_spike_time(spikes)))

compiler = EPropCompiler(
    example_timesteps=max_example_timesteps,
    losses="sparse_categorical_crossentropy",
    optimiser=Adam(5e-4),
    batch_size=BATCH_SIZE,
    feedback_type="random"
)

compiled_net = compiler.compile(network)

with compiled_net:
    start_time = perf_counter()

    metrics, _ = compiled_net.train(
        {input_pop: spikes},
        {output: labels},
        num_epochs=NUM_EPOCHS,
        shuffle=True,
        callbacks=["batch_progress_bar", Checkpoint(serialiser)]
    )

    end_time = perf_counter()

    print("p_max_in =", p_max_in)
    print("p_max_h =", p_max_h1)
    print("Accuracy =", 100 * metrics[output].result)
    print("Time =", end_time - start_time)
