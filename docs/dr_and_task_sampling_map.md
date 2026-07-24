# Domain randomization & task sampling map

Everything in `molmo_spaces` that injects variation into a generated episode, ordered by **when it fires**. Cross-checked against the MolmoBot (arXiv:2603.16861) and MolmoSpaces (arXiv:2602.11337) papers, 2026-07-24.

Rendered/interactive version: https://claude.ai/code/artifact/8cee2630-8cf9-4f9a-9c08-d9e5c7e0c539

---

## Fig 1 — The randomization lifecycle

Legend: **[ON]** = active by default · **[flag]** = opt-in via config flag, off by default.

### Once per scene — spec compile + `init_scene()` (`tasks/task_sampler.py:811`)

| Mechanism | Where | What it does |
|---|---|---|
| **[flag]** Robot speckle textures (`randomize_robot_textures`) | `robots/franka.py:28,163` | p = 0.7 per robot (`perturb_texture_probability`, `robot_configs.py:166`). Each flat-RGB material → PNG of base color + Gaussian noise + ~80 ellipse/rect blobs. **Franka family only — PiPER-X accepts the flag but ignores it.** |
| **[flag]** Empty material pool (`randomize_textures`) | `env/arena/randomization/texture.py` | ~200 placeholder materials/textures baked into `MjSpec`; must exist at compile time for per-episode texture swaps. |
| **[flag]** Door/handle joint DR (`enable_door_joint_randomization`) | `utils/mujoco_scene_utils.py` | Door stiffness 3–7, damping 8–12, frictionloss 8–12; handle stiffness 200–300, damping 80–120, frictionloss 40–60. |
| **[ON]** Seeded randomizer construction | `tasks/task_sampler.py:811-888` | One `base_seed` = task seed + 1, fans out: lighting = base, texture = base+1, dynamics = base+2 — independent `RandomState` streams. |

### Every episode — `randomize_scene()` (`tasks/task_sampler.py:978`)

| Mechanism | Where | What it does |
|---|---|---|
| **[flag]** Lighting (`randomize_lighting`) | `env/arena/randomization/lighting.py` | Position ±0.5 m, direction ±1 rad, ambient/diffuse/specular ±0.1, random on/off per light. (Light count/type/placement comes from the scene JSON at house build time, `housegen/builder.py` — the randomizer only perturbs what's there.) |
| **[flag]** Textures & materials (`randomize_textures` / `_all`) | `env/arena/randomization/texture.py` | `_all` → full `randomize()`, else `randomize_by_category()`. Swaps RGB maps from the THOR pool, RGBA ±0.2, specular ±0.1, shininess ±0.1 **around authored values** — BRDF identity is preserved (copied through swaps, `texture.py:1373`); `reflectance` never touched. 2% of episodes keep original textures. |
| **[flag]** Dynamics (`randomize_dynamics`) | `env/arena/randomization/dynamics.py` | Geom friction, mass, inertia, all ±20%. Applies to ALL bodies with joints — **robot links included**, no exclusion. Density is immutable in MuJoCo → mass is the knob. |
| **[ON]** Robot init pose noise | `pick_task_sampler.py:227` · `robot_configs.py:153` | Graduated per joint 0.025 → 0.175 rad (distal joints move more), Jacobian-weighted so TCP shift ≤ 10 cm. |
| **[ON]** Instruction sampling | `env/object_manager.py:1021-1163` | Referral expressions scored by CLIP image–text similarity, sampled via softmax, temperature 0.02. |

### Camera setup — `configs/camera_configs.py`

| Mechanism | Where | What it does |
|---|---|---|
| **[ON]** Per-camera pose & intrinsics noise | `configs/camera_configs.py` | Wrist: ±1.5 cm lateral / 0.5 cm vert / 2 cm depth, 8° roll, 4° pitch/yaw; FOV ±3–4°; shoulder 5 cm / 8°. Exocentric: spherical sampling around workspace center + lookat noise, full 360° azimuth. |
| **[ON]** Eval randomization level | `utils/eval_camera_randomization_utils.py` | Single 0–100 knob, piecewise-linear interpolation of every camera param; deterministic seed derived from episode identity. Eval only. |
| **[ON]** Fisheye warp (RB-Y1 head) | `utils/fisheye_warping.py:67` | Distortion params jittered every frame, factor 0.001. |

### Every control step — `apply_action_noise()` (`robots/abstract.py:180`)

| Mechanism | Where | What it does |
|---|---|---|
| **[ON — by default!]** Action noise (kinematic robot DR) | `configs/robot_configs.py:42-80` · `robots/abstract.py:85-245` | TCP-space noise proportional to the commanded delta: σ = 0.1·‖Δx‖ (zero command → zero noise), truncated-Gaussian, clipped ±2 cm position / ±0.1 rad (~5.7°) rotation, mapped to joint space via Jacobian pseudo-inverse and added to the commanded joint positions. Mobile base: planar σ = 0.1·‖Δp‖, clip ±2 cm / ±0.05 rad (RBY1 override). **The unnoised commands are saved separately → the dataset records clean actions while the sim executes noisy ones.** |

How the noise works (and why the arm stays on track even though cuRobo is open-loop): commands are **absolute** joint waypoints, so each new target is anchored to the nominal plan — previous deviation is swallowed into the position controller's error and servoed out; noise never integrates. Training pairs the (physically perturbed) observations with the clean absolute labels, which point back onto the nominal path from wherever the arm is — that's DART-style recovery supervision. It would NOT work with relative/delta actions.

### Post-hoc — offline re-render of saved episodes

| Mechanism | Where | What it does |
|---|---|---|
| **[opt-in tool]** Offline renderer DR | `renderer/offline_renderers/domain_randomization.py` | Reloads episode from `task_metadata.json` + state data and re-renders with randomized light intensity/color, shadow softness, texture pool swaps. OpenGL backend implements it; Omniverse doesn't. |
| **[not in this repo]** Image augmentation | MolmoBot training pipeline | ColorJitter, GaussianBlur, RandomPosterize, RandomSharpness, RandomGrayscale — applied at policy training time, not during datagen. |

---

## Fig 2 — Task sampling pipeline (PickTaskSampler family)

Where most of the dataset variation actually comes from. `tasks/pick_task_sampler.py` · `tasks/task_sampler.py`.

1. **Scene iteration** — ProcTHOR-10k `house_inds`, `samples_per_house` episodes per house before advancing.
2. **Receptacle choice** — from `receptacle_types`; up to 10 different receptacles tried.
3. **Pickup object selection** — scene objects filtered by grasp library (`filter_for_grasps`), Objaverse assets oversampled ×30 — or pick-from-set mode: external UIDs injected 0.15–0.5 m from reference objects.
4. **Object placement** — 6-DoF pose in annulus 0.1–0.8 m, orientation sampled for diverse approach angles, min separation 5 cm, rejection sampling ≤ 200 attempts. *Fails 200× → back to step 2, next receptacle.*
5. **Robot base placement** — annulus around receptacle (`base_pose_sampling_radius_range`), yaw ±45°, z-offset U(−0.30, +0.25) vs object height, occupancy-map safety radius, optional segmentation-based visibility check. *Fails 10× → back to step 4, replace object.*
6. **Init qpos noise + settle** — graduated joint noise (Fig 1), then `sim_settle_timesteps = 500`.
7. **Episode starts** — expert planner executes with pre-computed 6-DoF grasps, filtered by collision + IK feasibility.

Failure recovery wraps the whole pipeline: per-asset blacklist after 10 failures, bounded sequential failure counts, total attempts ≤ `samples_per_house × 6`.

---

## BRDF / reflectance — the fine print

- Every material has its own authored Phong scalars (`specular`, `shininess`, `reflectance`) — so BRDFs **differ across assets**, and sampling different objects into a scene gives BRDF diversity.
- The randomization **never re-samples BRDF identity**: texture swaps copy the original specular/shininess onto the new texture (`texture.py:1373`), then jitter ±0.1. A matte object never becomes glossy. `reflectance` and `emission` are never touched. No PBR (roughness/metallic) path exists.
- Consequence: reflectance variation in the data comes entirely from asset diversity, not from DR.

## Action noise demo (see the servo behavior live)

`scripts/viz/piper_x_action_noise_demo.py` — the cubes-in-cup preview runner with the action noise cranked up via env vars, plus a per-step measurement of the TCP-space gap between the clean commanded target and the actual arm pose (printed every 300 steps; bounded ⇒ jitter, not drift).

```bash
conda activate mlspaces
NOISE_SCALE=1.0 NOISE_POS_CAP=0.05 NOISE_ROT_CAP=0.3 \
  python scripts/viz/piper_x_action_noise_demo.py 21
```

Knobs (defaults in parentheses are the datagen values): `NOISE_SCALE` = `action_scale_factor` (0.1), `NOISE_POS_CAP` = TCP position clip in meters (0.02), `NOISE_ROT_CAP` = TCP rotation clip in rad (0.1). `VIEW=0` runs headless; the positional arg is the seed.

## PiPER-X status (cubes-in-cup)

- All visual/physics DR flags default to **off** — `PiperXCubesInCupDataGenConfig` inherits them unset. Episodes get sampling-level variation (cube count 1–4, cube/cup placement) plus the always-on TCP action noise, and nothing else.
- Fixed assets + no texture DR ⇒ **exactly one BRDF per surface across the whole dataset** — zero reflectance variation.
- `robots/piper_x.py` accepts `randomize_textures` in `add_robot_to_scene()` but never uses it — enabling `randomize_robot_textures` for PiPER-X silently does nothing (speckle DR is Franka-family only).
