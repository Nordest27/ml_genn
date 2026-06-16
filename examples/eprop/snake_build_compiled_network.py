

def build_compiled_network(connectivity_type="fixed"):
    global dale_l1_reg
    network = Network(default_params)
    hidden_layers = {}

    with network:

        # ================= POPULATIONS =================

        input_pop = Population(
            SpikeInput(max_spikes=INPUT_SIZE * WAIT_INC), INPUT_SHAPE
        )

        # hidden_layers["downsample"] = Population(
        #     AdaptiveLeakyIntegrateFire(
        #         v_thresh=0.61,
        #         tau_mem=10.0,
        #         tau_refrac=3.0,
        #         tau_adapt=300,
        #         beta=0.0
        #     ),
        #     DOWNSAMPLE_SHAPE
        # )

        # hidden_layers["unified"] = Population(
        #     AdaptiveLeakyIntegrateFire(
        #         v_thresh=0.61,
        #         tau_mem=10.0,
        #         tau_refrac=3.0,
        #         tau_adapt=300,
        #         beta=0.0
        #     ),
        #     UNIFIED_SHAPE
        # )

        hidden_layers["E"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61,
                tau_mem=10.0,
                tau_refrac=3.0,
                tau_adapt=300,
            ),
            HIDDEN_E_SHAPE
        )

        hidden_layers["I"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61,
                tau_mem=10.0,
                tau_refrac=3.0,
                tau_adapt=300,
            ),
            HIDDEN_I_SHAPE
        )

        
        hidden_layers["policy_field"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61,
                tau_mem=10.0,
                tau_refrac=3.0,
                tau_adapt=300,
            ),
            HIDDEN_I_SHAPE
        )
        policy = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"),
            NUM_OUTPUT
        )


        hidden_layers["value_field"] = Population(
            AdaptiveLeakyIntegrateFire(
                v_thresh=0.61,
                tau_mem=10.0,
                tau_refrac=3.0,
                tau_adapt=300,
            ),
            HIDDEN_I_SHAPE
        )
        value = Population(
            LeakyIntegrate(tau_mem=10.0, bias=0.0, readout="var"),
            1
        )

        # ================= INPUT → HIDDEN =================
        # Connection(
        #     input_pop,
        #     hidden_layers["downsample"],
        #     make_connectivity(
        #         connectivity_type=connectivity_type,
        #         src_shape=INPUT_SHAPE,
        #         p=CONN_P["I-H"], 
        #         sigma=SIGMA_IN, 
        #         desired_fan_in=DESIRED_FAN_IN_IN,
        #         sign=1
        #     ),
        #     exc_inh_sign=1
        # )

        for layer, prob, c_type in [
            (hidden_layers["I"], CONN_P["I-H"], connectivity_type),
            (hidden_layers["E"], CONN_P["I-H"], connectivity_type),
            # (policy, CONN_P["D-O"], "fixed"),
            # (value, CONN_P["D-O"], "fixed"),
        ]:
            Connection(
                # hidden_layers["downsample"],
                input_pop,
                layer,
                make_connectivity(
                    connectivity_type=c_type,
                    # src_shape=DOWNSAMPLE_SHAPE,
                    src_shape=INPUT_SHAPE,
                    p=prob,
                    sigma=SIGMA_IN,
                    desired_fan_in=DESIRED_FAN_IN_IN,
                    fan_in_scale=FAN_IN_SCALE_IN,
                    sign=1
                ),
                exc_inh_sign=1
            )

        # ================= EXCITATORY =================
        for layer, prob, c_type, sign in [
            (hidden_layers["I"], CONN_P["H-H"], connectivity_type, 1),
            (hidden_layers["E"], CONN_P["H-H"], connectivity_type, 1),
            (hidden_layers["policy_field"], CONN_P["H-H"], connectivity_type, 1),
            (hidden_layers["value_field"], CONN_P["H-H"], connectivity_type, 1),
            # (policy, CONN_P["H-P"], "fixed", 1),
            # (value, CONN_P["H-V"], "fixed", 1),
        ]:
            Connection(
                hidden_layers["E"],
                layer,
                make_connectivity(
                    connectivity_type=c_type,
                    src_shape=HIDDEN_E_SHAPE,
                    p=prob,
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H1,
                    sign=sign
                ),
                exc_inh_sign=sign
            )

        """
        for layer in [
            hidden_layers["I"],
            hidden_layers["E"],
            hidden_layers["downsample"],
        ]:
            Connection(
                hidden_layers["E"],
                layer,
                make_connectivity(
                    connectivity_type=connectivity_type,
                    src_shape=HIDDEN_E_SHAPE,
                    p=CONN_P["H-H"],
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H1,
                    sign=None
                ),
                target_var="ISynSigmaEps",
                exc_inh_sign=None
            )

            Connection(
                layer,
                hidden_layers["E"],
                make_connectivity(
                    connectivity_type=connectivity_type,
                    src_shape=HIDDEN_E_SHAPE,
                    p=CONN_P["H-H"],
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H1,
                    sign=1
                ),
                feedback_name="pert_eps_feedback",
                exc_inh_sign=1
            )
            Connection(
                layer,
                hidden_layers["E"],
                make_connectivity(
                    connectivity_type=connectivity_type,
                    src_shape=HIDDEN_E_SHAPE,
                    p=CONN_P["H-H"],
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H1,
                    sign=1
                ),
                feedback_name="pert_sigma_feedback",
                exc_inh_sign=1
            )"""

        # ================= INHIBITORY =================
        for layer, prob, c_type, sign in [
            (hidden_layers["I"], CONN_P["H-H"], connectivity_type, -1),
            (hidden_layers["E"], CONN_P["H-H"], connectivity_type, -1),
            (hidden_layers["policy_field"], CONN_P["H-H"], connectivity_type, -1),
            (hidden_layers["value_field"], CONN_P["H-H"], connectivity_type, -1),
            # (policy, CONN_P["H-P"], "fixed", -1),
            # (value, CONN_P["H-V"], "fixed", -1),
        ]:
            Connection(
                hidden_layers["I"],
                layer,
                make_connectivity(
                    connectivity_type=c_type,
                    src_shape=HIDDEN_I_SHAPE,
                    p=prob,
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H2,
                    sign=sign
                ),
                exc_inh_sign=sign
            )

        """
        for layer in [
            hidden_layers["I"],
            hidden_layers["E"],
            hidden_layers["downsample"]
        ]:
            Connection(
                hidden_layers["I"],
                layer,
                make_connectivity(
                    connectivity_type=connectivity_type,
                    src_shape=HIDDEN_I_SHAPE,
                    p=CONN_P["H-H"],
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H2,
                    sign=None
                ),
                target_var="ISynSigmaEps",
                exc_inh_sign=None
            )
            Connection(
                layer,
                hidden_layers["I"],
                make_connectivity(
                    connectivity_type=connectivity_type,
                    src_shape=HIDDEN_I_SHAPE,
                    p=CONN_P["H-H"],
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H2,
                    sign=-1
                ),
                feedback_name="pert_eps_feedback",
                exc_inh_sign=-1
            )
            Connection(
                layer,
                hidden_layers["I"],
                make_connectivity(
                    connectivity_type=connectivity_type,
                    src_shape=HIDDEN_I_SHAPE,
                    p=CONN_P["H-H"],
                    sigma=SIGMA_H,
                    desired_fan_in=DESIRED_FAN_IN_H2,
                    sign=-1
                ),
                feedback_name="pert_sigma_feedback",
                exc_inh_sign=-1
            )"""
        
        
        # ================= UNIFIED CONNECTIONS =================

        # Connection(
        #     input_pop,
        #     hidden_layers["unified"],
        #     make_connectivity(
        #         connectivity_type=connectivity_type,
        #         src_shape=INPUT_SHAPE,
        #         p=CONN_P["I-H"], 
        #         sigma=SIGMA_IN, 
        #         desired_fan_in=DESIRED_FAN_IN_IN
        #     )
        # )

        # Connection(
        #     hidden_layers["unified"],
        #     hidden_layers["unified"],
        #     make_connectivity(
        #         connectivity_type=connectivity_type,
        #         src_shape=UNIFIED_SHAPE,
        #         p=CONN_P["H-H"], 
        #         sigma=SIGMA_H, 
        #         desired_fan_in=DESIRED_FAN_IN_H1
        #     )
        # )
 
        # Connection(
        #     hidden_layers["unified"],
        #     hidden_layers["policy_field"],
        #     make_connectivity(
        #         connectivity_type=connectivity_type,
        #         src_shape=UNIFIED_SHAPE,
        #         p=0.5, 
        #         sigma=SIGMA_H, 
        #         desired_fan_in=DESIRED_FAN_IN_H1
        #     )
        # )
        
        # Connection(
        #     hidden_layers["unified"],
        #     hidden_layers["value_field"],
        #     make_connectivity(
        #         connectivity_type=connectivity_type,
        #         src_shape=UNIFIED_SHAPE,
        #         p=0.5, 
        #         sigma=SIGMA_H, 
        #         desired_fan_in=DESIRED_FAN_IN_H1
        #     )
        # )


        # ================= OUTPUT CONNECTIONS =================
        Connection(
            hidden_layers["policy_field"],
            policy,
            make_connectivity(
                connectivity_type="fixed",
                src_shape=HIDDEN_I_SHAPE,
                p=0.99999,
                sigma=SIGMA_H,
                desired_fan_in=DESIRED_FAN_IN_H2,
                sign=None
            ),
            exc_inh_sign=None
        )
        Connection(
            hidden_layers["value_field"],
            value,
            make_connectivity(
                connectivity_type="fixed",
                src_shape=HIDDEN_I_SHAPE,
                p=0.99999,
                sigma=SIGMA_H,
                desired_fan_in=DESIRED_FAN_IN_H2,
                sign=None
            ),
            exc_inh_sign=None
        )

        # ================= FEEDBACK CONNECTIONS =================
        Connection(
            hidden_layers["policy_field"], policy,
            FixedProbability(0.99999, Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
            feedback_name="policy_feedback",
            exc_inh_sign=None
        )
        Connection(
            hidden_layers["value_field"], value,
            FixedProbability(0.99999, Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
            feedback_name="value_feedback",
            exc_inh_sign=None
        )

        Connection(
            policy, value, Dense(weight=1.0),
            feedback_name="tde_transport"
        )
        for h_name, hidden_layer in hidden_layers.items():
            Connection(
                hidden_layer, value, Dense(weight=1.0),
                feedback_name="tde_transport"
            )
            if "field" in h_name:
                continue
            sign = None
            if hidden_layer == hidden_layers.get("E"):
                sign = 1
            elif hidden_layer == hidden_layers.get("I"):
                sign = -1

            Connection(
                hidden_layer, policy,
                FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                feedback_name="policy_feedback",
                exc_inh_sign=sign
            )
            # Connection(
            #     hidden_layer, policy,
            #     FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
            #     feedback_name="policy_regularisation",
            #     exc_inh_sign=sign
            # )
            Connection(
                hidden_layer, value,
                FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
                feedback_name="value_feedback",
                exc_inh_sign=sign
            )
            # Connection(
            #     hidden_layer, value,
            #     FixedProbability(CONN_P["F"], Normal(sd=1.0 / np.sqrt(NUM_OUTPUT))),
            #     feedback_name="value_regularisation",
            #     exc_inh_sign=sign
            # )
            

    # ================= COMPILER =================
    dale_l1_reg = 0.0001/np.sqrt(DESIRED_FAN_IN_IN)
    if CONNECTIVITY_TYPE == "fixed":
        dale_l1_reg = 0.01/np.sqrt(max(INPUT_SIZE, NUM_HIDDEN_E))
    dale_l1_reg = 0
    print("L1 reg strength:", dale_l1_reg)
    compiler = EPropCompiler(
        example_timesteps=1,
        losses={
            policy: "mean_square_error" # "sparse_categorical_crossentropy" 
                if not GAUSSIAN_TRACE_POLICY else 
                "mean_square_error",
            value: "mean_square_error"
        },
        optimiser=Adam(1e-4),#, soft_grad_clip=10), 
        # optimiser=Adam(1e-4, clamp_grad=(-5.0, 5.0)),
        c_reg=1e-2,
        batch_size=1,
        kernel_profiling=KERNEL_PROFILING,
        feedback_type="random",
        reward_decay=reward_decay,
        gamma=gamma,
        td_lambda=td_lambda,
        train_output_bias=False,
        reset_time_between_batches=False,
        entropy_coeff=entropy_coeff,
        entropy_coeff_decay=entropy_decay,
        entropy_coeff_min=entropy_coeff_min,
        dale_rewiring_l1_strength=dale_l1_reg,
        policy_heads={
            policy: PolicyTypes.GENERIC 
            if not GAUSSIAN_TRACE_POLICY 
            else PolicyTypes.GAUSSIAN_TRACE, 
        },
        value_head=value
    )

    if CHECKPOINT_BOARD_SIZE is not None:
        network.load((CHECKPOINT_BOARD_SIZE,), serialiser)

    compiled_net = compiler.compile(network)

    return compiled_net, network, input_pop, hidden_layers, policy, value
