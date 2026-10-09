import logging
import numpy as np

from copy import deepcopy
from typing import Sequence
from pygenn import SynapseMatrixType, create_var_ref, create_wu_var_ref

from .. import Compiler
from ..compiled_training_network import CompiledTrainingNetwork
from ..deep_r import RewiringRecord, add_deep_r
from ..dale_rewiring import add_dale_rewiring
from ..compiler import create_reset_custom_update
from ... import Connection, Population, Network
from ...callbacks import (BatchProgressBar, CustomUpdateOnBatchBegin,
                          CustomUpdateOnBatchEnd, CustomUpdateOnEpochBegin,
                          CustomUpdateOnTimestepEnd)
from ...communicators import Communicator
from ...losses import Loss, SparseCategoricalCrossentropy, default_losses
from ...metrics import MetricsType
from ...neurons import (AdaptiveLeakyIntegrateFire, Input, LeakyIntegrate,
                        LeakyIntegrateFire, LeakyIntegrateFireInput)
from ...optimisers import Optimiser, default_optimisers
from ...synapses import Delta
from ...utils.callback_list import CallbackList
from ...utils.model import CustomUpdateModel, NeuronModel, SynapseModel, WeightUpdateModel
from ...utils.module import get_object, get_object_mapping
from ...utils.network import get_underlying_conn
from ...utils.snippet import ConnectivitySnippet
from ...utils.value import is_value_constant

from .variants import (FeedbackType, PolicyType, HiddenRuleConfig, NoisePlacement,
                       get_hidden_rule, as_policy_type, rule_for_population)
from .models import GRADIENT_BATCH_REDUCE_MODEL
from . import neuron_logic
from . import connection_logic as conn_logic

logger = logging.getLogger(__name__)

default_params = {
    AdaptiveLeakyIntegrateFire: {"relative_reset": True,
                                 "integrate_during_refrac": True},
    LeakyIntegrate: {"scale_i": False},
    LeakyIntegrateFire: {"relative_reset": True,
                         "integrate_during_refrac": True,
                         "scale_i": False},
    LeakyIntegrateFireInput: {"relative_reset": True,
                              "integrate_during_refrac": True,
                              "scale_i": False}}


def _has_connection_to_output(pop):
    for c in pop.outgoing_connections:
        if c().target().neuron.readout is not None:
            return True
    return False


def _has_feedback_from_output(pop):
    for c in pop.outgoing_connections:
        if c().is_feedback:
            return True
    return False


class CompileState:
    def __init__(self, losses, readouts):
        self.losses = get_object_mapping(losses, readouts,
                                         Loss, "Loss", default_losses)
        self._tau_mem = None
        self._tau_adapt = None

        self.feedback_connections = []
        self.sigma_eps_connections = []
        self.tde_transport_connections = []
        self.policy_reward_connections = []
        self.pert_eps_transport_connections = []
        self.backprop_connections = []           # hidden -> hidden synapses sending the backward signal
        self.value_feedback_connections = []
        self.policy_feedback_connections = []
        self.value_regularisation_connections = []
        self.policy_regularisation_connections = []

        self.weight_optimiser_connections = []
        self.bias_optimiser_populations = []
        self.softmax_populations = []
        self._neuron_reset_vars = {}
        self.checkpoint_connection_vars = []
        self.checkpoint_population_vars = []

    def add_neuron_readout_reset_vars(self, pop):
        reset_vars = pop.neuron.readout.reset_vars
        if len(reset_vars) > 0:
            assert pop not in self._neuron_reset_vars
            self._neuron_reset_vars[pop] = reset_vars

    def create_reset_custom_updates(self, compiler, genn_model, neuron_pops):
        for i, (pop, reset_vars) in enumerate(self._neuron_reset_vars.items()):
            model = create_reset_custom_update(
                reset_vars,
                lambda name: create_var_ref(neuron_pops[pop], name))
            compiler.add_custom_update(genn_model, model,
                                       "Reset", f"CUResetNeuron{i}")

    @property
    def is_reset_custom_update_required(self):
        return len(self._neuron_reset_vars) > 0

    @property
    def tau_mem(self):
        assert self._tau_mem is not None
        return self._tau_mem

    @tau_mem.setter
    def tau_mem(self, tau_mem):
        if self._tau_mem is None:
            self._tau_mem = tau_mem
        if self._tau_mem != tau_mem:
            raise NotImplementedError("E-prop compiler doesn't "
                                      "support neurons with "
                                      "different time constants")

    @property
    def tau_adapt(self):
        assert self._tau_adapt is not None
        return self._tau_adapt

    @tau_adapt.setter
    def tau_adapt(self, tau_adapt):
        if self._tau_adapt is None:
            self._tau_adapt = tau_adapt
        if self._tau_adapt != tau_adapt:
            raise NotImplementedError("E-prop compiler doesn't "
                                      "support neurons with "
                                      "different time constants")


class EPropCompiler(Compiler):
    """Compiler for training models using e-prop [Bellec2020]_.

    The e-prop compiler supports :class:`ml_genn.neurons.LeakyIntegrateFire` and
    :class:`ml_genn.neurons.AdaptiveLeakyIntegrateFire` hidden neuron models; and
    :class:`ml_genn.losses.SparseCategoricalCrossentropy` loss functions for classification
    and :class:`ml_genn.losses.MeanSquareError` for regression.

    e-prop is derived from Real-Time Recurrent Learning (RTRL) so does not require a
    backward pass meaning that its memory overhead does not scale with sequence length.
    However, e-prop requires a per-connection eligibility trace meaning that it is
    incompatible with connectivity like convolutions with shared weights. Furthermore,
    because each connection has to be updated every timestep, training performance is not
    improved by sparse activations.

    Args:
        example_timesteps:          How many timesteps each example will be
                                    presented to the network for
        losses:                     Either a dictionary mapping loss functions
                                    to output populations or a single loss
                                    function to apply to all outputs
        optimiser:                  Optimiser to use when applying weights
        tau_reg:                    Time constant with which hidden neuron
                                    spike trains are filtered to obtain the
                                    firing rate used for regularisation [ms]
        c_reg:                      Regularisation strength
        f_target:                   Target hidden neuron firing rate used for
                                    regularisation [Hz]
        train_output_bias:          Should output neuron biases be trained?
        dt:                         Simulation timestep [ms]
        batch_size:                 What batch size should be used for
                                    training? In our experience, e-prop works
                                    well with very large batch sizes (512)
        rng_seed:                   What value should GeNN's GPU RNG be seeded
                                    with? This is used for all GPU randomness
                                    e.g. weight initialisation and Poisson
                                    spike train generation
        kernel_profiling:           Should GeNN record the time spent in each
                                    GPU kernel? These values can be extracted
                                    directly from the GeNN model which can be
                                    accessed via the ``genn_model`` property
                                    of the compiled model.
        reset_time_between_batches: Should time be reset to zero at the start
                                    of each example or allowed to run
                                    continously?
        communicator:               Communicator used for inter-process
                                    communications when training across
                                    multiple GPUs.
        feedback_type:              One of "symmetric", "random", "adaptive"
                                    (see :class:`.variants.FeedbackType`)
        policy_heads:                Mapping from readout Population to a
                                    :class:`.variants.PolicyType`, required
                                    when running the RL variant.
        value_head:                  The value-head readout Population,
                                    required when running the RL variant.
        hidden_rule:                 Hidden-layer rule of the RL/TD(lambda)
                                    variant: a :class:`.variants.HiddenRuleConfig`
                                    or a preset name from
                                    :data:`.variants.PRESETS` ("proposed",
                                    "original", "drift_only", "gradient_only",
                                    "unbiased", "eprop", ...). Ignored outside
                                    the RL variant.
        value_feedback_ret_e:        RetE of the value feedback connections, which
                                    deliver VE to the hidden neurons (used only by
                                    the e-prop term). 0.0: VE = B^V, the critic
                                    part of e-prop's learning signal (Bellec et al.
                                    2020); 1.0: VE = B^V * TD error, as in the
                                    original code, which makes the critic part
                                    ~ delta^2 B^V e (a biased push, not a gradient).
        optimise_feedback:           Optimise adaptive feedback connections
                                    (False reproduces the original
                                    implementation, where they are not).
        value_reg:                   Weight of the value head's smoothness
                                    regulariser (0.0 as in the original
                                    implementation).
    """
    def __init__(self, example_timesteps: int, losses, optimiser="adam",
                 tau_reg: float = 500.0, c_reg: float = 0.001,
                 f_target: float = 10.0, train_output_bias: bool = True,
                 dt: float = 1.0, batch_size: int = 1,
                 rng_seed: int = 0, kernel_profiling: bool = False,
                 reset_time_between_batches: bool = True,
                 communicator: Communicator = None,
                 deep_r_conns: Sequence = [],
                 deep_r_l1_strength: float = 0.01,
                 deep_r_record_rewirings={},
                 dale_rewiring_l1_strength: float = 0,
                 feedback_type: str = "symmetric",
                 reward_decay: float = 0.9,
                 gamma: float = None,
                 td_lambda: float = None,
                 entropy_coeff: float = 1e-4,
                 entropy_coeff_decay: float = 0.99,
                 entropy_coeff_min: float = 1e-6,
                 policy_heads: Population = None,
                 value_head: Population = None,
                 hidden_rule="proposed",
                 value_feedback_ret_e: float = 0.0,
                 optimise_feedback: bool = False,
                 value_reg: float = 0.0,
                 **genn_kwargs):
        supported_matrix_types = [SynapseMatrixType.SPARSE,
                                  SynapseMatrixType.DENSE]
        super(EPropCompiler, self).__init__(supported_matrix_types, dt,
                                            batch_size, rng_seed,
                                            kernel_profiling,
                                            communicator,
                                            **genn_kwargs)
        self.example_timesteps = example_timesteps
        self.losses = losses
        self._optimiser = get_object(optimiser, Optimiser, "Optimiser",
                                     default_optimisers)
        self.reward_decay = reward_decay
        self.tau_reg = tau_reg
        self.c_reg = c_reg
        self.f_target = f_target
        self.train_output_bias = train_output_bias
        self.reset_time_between_batches = reset_time_between_batches
        self.deep_r_conns = set(get_underlying_conn(c) for c in deep_r_conns)
        self.deep_r_l1_strength = deep_r_l1_strength
        self.deep_r_record_rewirings = {get_underlying_conn(c): k
                                        for c, k in deep_r_record_rewirings.items()}

        self.dale_rewiring_l1_strength = dale_rewiring_l1_strength
        self.feedback_type = feedback_type

        self.gamma = gamma
        self.td_lambda = td_lambda
        self.gamma_lambda = None
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff_decay
        self.entropy_coeff_min = entropy_coeff_min
        self.policy_heads = ({p: as_policy_type(t) for p, t in policy_heads.items()}
                             if policy_heads is not None else None)
        self.value_head = value_head
        self.hidden_rule = get_hidden_rule(hidden_rule)
        self.value_feedback_ret_e = value_feedback_ret_e
        self.optimise_feedback = optimise_feedback
        self.value_reg = value_reg
        self._configure_rl(gamma, td_lambda, self.policy_heads, value_head)

    def _configure_rl(self, gamma, td_lambda, policy_heads, value_head):
        """Validate and derive RL-specific configuration. Either both gamma
        and td_lambda (plus heads) are given, activating the RL/TD(lambda)
        variant, or neither is, keeping the compiler in supervised mode.
        """
        rl_requested = (gamma is not None) and (td_lambda is not None)
        heads_given = (policy_heads is not None) or (value_head is not None)

        if rl_requested:
            self.gamma_lambda = gamma * td_lambda
            if policy_heads is None or value_head is None:
                raise ValueError("Policy and value heads must be specified "
                                 "when activating the RL Eprop variant")
        elif heads_given:
            raise ValueError("Neither policy nor value heads must be "
                             "specified when not running the RL Eprop variant")

    # ------------------------------------------------------------------
    # Compiler hooks
    # ------------------------------------------------------------------

    def pre_compile(self, network: Network, genn_model, **kwargs) -> CompileState:
        readouts = [p for p in network.populations
                    if p.neuron.readout is not None]
        return CompileState(self.losses, readouts)

    def build_neuron_model(self, pop: Population, model: NeuronModel,
                           compile_state: CompileState) -> NeuronModel:
        model_copy = deepcopy(model)

        if pop.neuron.readout is not None:
            self._build_output_neuron_model(pop, model_copy, compile_state)
        elif not isinstance(pop.neuron, Input):
            self._build_hidden_neuron_model(pop, model_copy, compile_state)
        else:
            neuron_logic.add_input_noise_code(model_copy)

        return model_copy

    def build_synapse_model(self, conn: Connection, model: SynapseModel,
                            compile_state: CompileState) -> SynapseModel:
        if not isinstance(conn.synapse, Delta):
            raise NotImplementedError("E-prop compiler only "
                                      "supports Delta synapses")
        return model

    def build_weight_update_model(self, conn: Connection,
                                  connect_snippet: ConnectivitySnippet,
                                  compile_state: CompileState) -> WeightUpdateModel:
        if not is_value_constant(connect_snippet.delay):
            raise NotImplementedError("E-prop compiler only "
                                      "support heterogeneous delays")

        alpha = np.exp(-self.dt / compile_state.tau_mem)
        target_pop = conn.target()
        target_neuron = target_pop.neuron

        if conn.is_feedback:
            wum = conn_logic.build_feedback_wum(
                conn, connect_snippet, self, compile_state, target_pop, alpha)
        elif isinstance(target_neuron, (LeakyIntegrateFire, AdaptiveLeakyIntegrateFire)):
            wum = conn_logic.build_hidden_wum(
                conn, connect_snippet, self, compile_state, target_neuron, alpha)
        elif target_neuron.readout is not None:
            wum = conn_logic.build_output_wum(
                conn, connect_snippet, self, compile_state, target_pop, alpha)
        else:
            raise NotImplementedError(
                f"E-prop compiler doesn't support connections targeting "
                f"{type(target_neuron).__name__} neurons")

        compile_state.checkpoint_connection_vars.append((conn, "g"))
        # adaptive feedback connections have DeltaG; they are optimised only on request
        # (the original implementation never optimised them)
        if not conn.is_feedback or (self.optimise_feedback and
                                    "DeltaG" in [v[0] for v in wum.model["vars"]]):
            compile_state.weight_optimiser_connections.append(conn)

        return wum

    # ------------------------------------------------------------------
    # Neuron-model construction (split out of build_neuron_model)
    # ------------------------------------------------------------------

    def _build_output_neuron_model(self, pop, model_copy, compile_state):
        loss = compile_state.losses[pop]

        if isinstance(loss, SparseCategoricalCrossentropy):
            neuron_logic.add_softmax_output_var(model_copy, compile_state, pop)

        model_copy = pop.neuron.readout.add_readout_logic(
            model_copy, example_timesteps=self.example_timesteps, dt=self.dt)
        compile_state.add_neuron_readout_reset_vars(pop)

        model_copy.add_var("E", "scalar", 0.0)
        loss.add_to_neuron(model_copy, pop.shape, self.batch_size,
                           self.example_timesteps)

        if self.gamma_lambda is not None:
            neuron_logic.add_rl_output_head_code(model_copy, pop, self)
        else:
            neuron_logic.add_supervised_error_code(model_copy)

        if self.train_output_bias:
            neuron_logic.add_output_bias_training_code(model_copy, compile_state, pop)

    def _build_hidden_neuron_model(self, pop, model_copy, compile_state):
        neuron_logic.add_hidden_feedback_code(model_copy)

        if self.gamma_lambda is not None:
            neuron_logic.add_hidden_rl_input_refs(model_copy, backprop=self.hidden_rule.backprop != 0)
            pop_rule = rule_for_population(self.hidden_rule, pop, self.policy_heads, self.value_head)
            if (pop_rule.noise is NoisePlacement.NODE
                    and isinstance(pop.neuron, AdaptiveLeakyIntegrateFire)):
                neuron_logic.enable_node_noise(model_copy)

        if not isinstance(pop.neuron, (AdaptiveLeakyIntegrateFire, LeakyIntegrateFire)):
            raise NotImplementedError(f"E-prop compiler doesn't support "
                                      f"{type(pop.neuron).__name__} neurons")

        if not pop.neuron.integrate_during_refrac:
            logger.warning("E-prop learning works best with (A)LIF "
                           "neurons which continue to integrate "
                           "during their refractory period")
        if not pop.neuron.relative_reset:
            logger.warning("E-prop learning works best with (A)LIF "
                           "neurons with a relative reset mechanism")

        compile_state.tau_mem = pop.neuron.tau_mem
        if isinstance(pop.neuron, AdaptiveLeakyIntegrateFire):
            compile_state.tau_adapt = pop.neuron.tau_adapt

    # ------------------------------------------------------------------
    # Network assembly
    # ------------------------------------------------------------------

    def create_compiled_network(self, genn_model, neuron_populations: dict,
                                connection_populations: dict,
                                compile_state: CompileState) -> CompiledTrainingNetwork:
        genn_model.fuse_postsynaptic_models = True
        genn_model.fuse_pre_post_weight_update_models = True

        self._wire_feedback_targets(connection_populations, compile_state)

        optimiser_custom_updates = []
        deep_r_record_rewirings_ccus = []
        dale_rewiring_required = self._add_weight_optimisers(
            genn_model, connection_populations, compile_state, optimiser_custom_updates,
            deep_r_record_rewirings_ccus)

        self._add_bias_optimisers(genn_model, neuron_populations, compile_state,
                                  optimiser_custom_updates)

        for p, o, s in compile_state.softmax_populations:
            genn_pop = neuron_populations[p]
            self.add_softmax_custom_updates(genn_model, genn_pop, o, s)

        compile_state.create_reset_custom_updates(self, genn_model, neuron_populations)

        base_train_callbacks, base_validate_callbacks = self._build_callbacks(
            compile_state, optimiser_custom_updates, deep_r_record_rewirings_ccus,
            dale_rewiring_required)

        optimisers = []
        if len(optimiser_custom_updates) > 0:
            optimisers.append((self._optimiser, optimiser_custom_updates))

        return CompiledTrainingNetwork(
            genn_model, neuron_populations, connection_populations,
            self.communicator, compile_state.losses,
            self.example_timesteps, base_train_callbacks,
            base_validate_callbacks, optimisers,
            compile_state.checkpoint_connection_vars,
            compile_state.checkpoint_population_vars, self.reset_time_between_batches)

    def _wire_feedback_targets(self, connection_populations, compile_state):
        """Point each feedback-carrying connection's pre/post target var at
        the correct ISyn* input on the receiving neuron.
        """
        for c in compile_state.feedback_connections:
            connection_populations[c].pre_target_var = "ISynFeedback"

        if self.gamma_lambda is None:
            return

        target_map = [
            (compile_state.sigma_eps_connections, "post_target_var", "ISynSigmaEps"),
            (compile_state.sigma_eps_connections, "pre_target_var", "ISynPertEps"),
            (compile_state.tde_transport_connections, "pre_target_var", "ISynTdE"),
            (compile_state.policy_feedback_connections, "pre_target_var", "ISynPolicyReward"),
            (compile_state.pert_eps_transport_connections, "pre_target_var", "ISynPertEps"),
            (compile_state.policy_feedback_connections, "pre_target_var", "ISynPolicyGradient"),
            (compile_state.policy_regularisation_connections, "pre_target_var", "ISynPolicyRegularisation"),
            (compile_state.value_feedback_connections, "pre_target_var", "ISynValueError"),
            (compile_state.value_regularisation_connections, "pre_target_var", "ISynValueRegularisation"),
            (compile_state.backprop_connections, "pre_target_var", "ISynBack"),
        ]
        for conns, attr, target_var in target_map:
            for c in conns:
                setattr(connection_populations[c], attr, target_var)

    def _add_weight_optimisers(self, genn_model, connection_populations, compile_state,
                               optimiser_custom_updates, deep_r_record_rewirings_ccus):
        """Add Deep-R / Dale rewiring infrastructure and an optimiser custom
        update for every weight that needs one. Returns whether Dale
        rewiring was used anywhere.
        """
        dale_rewiring_required = False
        for i, c in enumerate(compile_state.weight_optimiser_connections):
            genn_pop = connection_populations[c]
            dale_sign = c.exc_inh_sign
            delta_g_var_ref = create_wu_var_ref(genn_pop, "DeltaG")
            weight_var_ref = create_wu_var_ref(genn_pop, "g")

            if c in self.deep_r_conns:
                deep_r_2_ccu = add_deep_r(genn_pop, genn_model,
                                          self, self.deep_r_l1_strength,
                                          delta_g_var_ref, weight_var_ref)
                if c in self.deep_r_record_rewirings:
                    deep_r_record_rewirings_ccus.append(
                        (deep_r_2_ccu, self.deep_r_record_rewirings[c]))

            if dale_sign is not None:
                dale_rewiring_required = True
                add_dale_rewiring(
                    synapse_group=genn_pop, genn_model=genn_model,
                    compiler=self, l1_strength=self.dale_rewiring_l1_strength,
                    sign=dale_sign, weight_var_ref=weight_var_ref)

            optimiser_custom_updates.append(
                self._create_optimiser_custom_update(
                    f"Weight{i}", weight_var_ref, delta_g_var_ref,
                    genn_model, True))

        return dale_rewiring_required

    def _add_bias_optimisers(self, genn_model, neuron_populations, compile_state,
                             optimiser_custom_updates):
        for i, p in enumerate(compile_state.bias_optimiser_populations):
            genn_pop = neuron_populations[p]
            optimiser_custom_updates.append(
                self._create_optimiser_custom_update(
                    f"Bias{i}", create_var_ref(genn_pop, "Bias"),
                    create_var_ref(genn_pop, "DeltaBias"),
                    genn_model, False))

    def _build_callbacks(self, compile_state, optimiser_custom_updates,
                         deep_r_record_rewirings_ccus, dale_rewiring_required):
        train_callbacks = []
        validate_callbacks = []
        deep_r_required = (len(self.deep_r_conns) > 0)

        if deep_r_required and self.deep_r_l1_strength > 0.0:
            train_callbacks.append(CustomUpdateOnBatchEnd("DeepRL1"))
        if dale_rewiring_required and self.dale_rewiring_l1_strength > 0.0:
            train_callbacks.append(CustomUpdateOnBatchEnd("DaleRL1"))

        if len(optimiser_custom_updates) > 0:
            if self.full_batch_size > 1:
                train_callbacks.append(CustomUpdateOnBatchEnd("GradientBatchReduce"))
            train_callbacks.append(CustomUpdateOnBatchEnd("GradientLearn"))

        if compile_state.is_reset_custom_update_required:
            train_callbacks.append(CustomUpdateOnBatchBegin("Reset"))
            validate_callbacks.append(CustomUpdateOnBatchBegin("Reset"))

        if deep_r_required:
            train_callbacks.append(CustomUpdateOnEpochBegin("DeepRInit", lambda e: e == 0))
            train_callbacks.append(CustomUpdateOnBatchEnd("DeepR1"))
            train_callbacks.append(CustomUpdateOnBatchEnd("DeepR2"))

        for c, k in deep_r_record_rewirings_ccus:
            train_callbacks.append(RewiringRecord(c, k))

        if len(compile_state.softmax_populations) > 0:
            for cb_name in ("Softmax1", "Softmax2", "Softmax3"):
                train_callbacks.append(CustomUpdateOnTimestepEnd(cb_name))
                validate_callbacks.append(CustomUpdateOnTimestepEnd(cb_name))

        if dale_rewiring_required:
            train_callbacks.append(CustomUpdateOnEpochBegin("DaleInit", lambda e: e == 0))
            train_callbacks.append(CustomUpdateOnBatchEnd("DalePrune"))
            train_callbacks.append(CustomUpdateOnBatchEnd("DaleRewire"))

        return train_callbacks, validate_callbacks

    def _create_optimiser_custom_update(self, name_suffix, var_ref,
                                        gradient_ref, genn_model, wu):
        if self.full_batch_size > 1:
            reduction_optimiser_model = CustomUpdateModel(
                GRADIENT_BATCH_REDUCE_MODEL, {}, {"ReducedGradient": 0.0},
                {"Gradient": gradient_ref})

            genn_reduction = self.add_custom_update(
                genn_model, reduction_optimiser_model,
                "GradientBatchReduce", "CUBatchReduce" + name_suffix)
            reduced_gradient = (create_wu_var_ref(genn_reduction, "ReducedGradient") if wu
                                else create_var_ref(genn_reduction, "ReducedGradient"))
            optimiser_model = self._optimiser.get_model(reduced_gradient, var_ref, False, None)
        else:
            optimiser_model = self._optimiser.get_model(gradient_ref, var_ref, True, None)

        return self.add_custom_update(genn_model, optimiser_model,
                                      "GradientLearn", "CUGradientLearn" + name_suffix)
