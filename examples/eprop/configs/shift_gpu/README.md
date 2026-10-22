# Relays with displaced centres (lateral transport through depth)

`to_next.centre_shift = r` (local mode): every target's Gaussian kernel into the next layer is centred on a cell up
to r cells away, from one displacement field per layer-to-layer step shared by all its populations (E, I, slow), so
neighbouring E and I neurons are displaced together. Each stage stays a local, averaging relay, but the topography
is reorganised, so the recurrence of every layer integrates different parts of the frame.

All configs = c298 (the best full-size relay, arch_search_full_v6_l5_relay) with its calibration, ff_centre_shift
0 / 2 / 3, and either your rule (proposed + decorrelation, as arch_gpu/d5_narrow_proposed_decorr1) or the reservoir.
The learning gain per architecture is rule minus reservoir. The displacement field is drawn from the run's seed.

Untrained, full size (current frame at the readout / field memory lag 1 / dead E neurons in L4, L5):
shift 0: 0.74 / 0.38 / 42%, 47%   shift 2: 0.72 / 0.41 / 19%, 19%   shift 3: 0.71 / 0.40 / 12%, 17%
Lateral transport (input decoded at its own position vs from the whole layer, value field): shift 0 0.66 vs 0.76,
shift 3 0.49 vs 0.70: the information moves sideways rather than disappearing.
