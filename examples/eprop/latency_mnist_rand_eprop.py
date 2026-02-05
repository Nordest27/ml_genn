import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import InputLayer, Layer, Network, Population, Connection
from ml_genn.callbacks import Checkpoint
from ml_genn.compilers import EPropCompiler, InferenceCompiler, RandEPropCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, AdaptiveLeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.optimisers import Adam

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data)

from ml_genn.compilers.eprop_compiler import default_params

NUM_INPUT = 784
NUM_HIDDEN_1 = 128
NUM_HIDDEN_2 = 128
NUM_OUTPUT = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
TRAIN = True
KERNEL_PROFILING = False

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = log_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    20.0, 51)

serialiser = Numpy("latency_mnist_rand_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * calc_max_spikes(spikes)),
                                  NUM_INPUT)
    hidden_1 = Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                           tau_refrac=5.0),
                        NUM_HIDDEN_1)
    hidden_2 = Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                           tau_refrac=5.0),
                        NUM_HIDDEN_2)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="sum_var"),
                        NUM_OUTPUT)
    
    # Connections
    Connection(input,  hidden_1, Dense(Normal(sd=1.0 / np.sqrt(NUM_INPUT))))
    Connection(hidden_1, hidden_2, FixedProbability(0.5, (Normal(sd=1.0 / np.sqrt(NUM_HIDDEN_1)))))
    Connection(hidden_2, output, Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN_2))))
    
    # UNCOMMENT FOR SYMMETRIC EPROP COMPARISON
    # Connection(hidden_1, output, FixedProbability(0.0001, Normal(sd=1.0 / np.sqrt(NUM_HIDDEN_1))))

    # Random feedback matrices (COMMENT THESE LINES TO COMPARE WITH SYMMETRIC EPROP)
    Connection(hidden_1, output, Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))), feedback_name="f1")
    Connection(hidden_2, output, Dense(Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))), feedback_name="f2")

max_example_timesteps = int(np.ceil(calc_latest_spike_time(spikes)))
if TRAIN:
    compiler = EPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             optimiser=Adam(5e-4),
                             batch_size=BATCH_SIZE,
                             kernel_profiling=KERNEL_PROFILING,
                             feedback_type="random")
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", Checkpoint(serialiser)]
        metrics, _  = compiled_net.train({input: spikes},
                                         {output: labels},
                                         num_epochs=NUM_EPOCHS, shuffle=True,
                                         callbacks=callbacks)
        compiled_net.save_connectivity((NUM_EPOCHS - 1,), serialiser)
        
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")

        if KERNEL_PROFILING:
            print(f"Neuron update time = {compiled_net.genn_model.neuron_update_time}")
            print(f"Presynaptic update time = {compiled_net.genn_model.presynaptic_update_time}")
            print(f"Synapse dynamics time = {compiled_net.genn_model.synapse_dynamics_time}")
            print(f"Gradient batch reduce time = {compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')}")
            print(f"Gradient learn time = {compiled_net.genn_model.get_custom_update_time('GradientLearn')}")
            print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")
            print(f"Softmax1 time = {compiled_net.genn_model.get_custom_update_time('Softmax1')}")
            print(f"Softmax2 time = {compiled_net.genn_model.get_custom_update_time('Softmax2')}")
            print(f"Softmax3 time = {compiled_net.genn_model.get_custom_update_time('Softmax3')}")
else:
    # Load network state from final checkpoint
    network.load((NUM_EPOCHS - 1,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                 batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels})
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
