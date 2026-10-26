# Historical IK planner comparison for the five-cube simulation

The online Cartesian controller is now the default when running `simulation/line_assembly.py`. The `--planner` flag is an alias for `--legacy-planner` and explicitly runs the historical joint-trajectory controller discussed below.

The same ten scattered scenes (`--seed 0` through `--seed 9`) and the fixed scene were used for each full-rollout comparison. A pass requires all five cubes to meet the existing 1 cm XY and 5° orientation limits, with no gripper/table contact, cube/table contact during transfer, or gripper contact with another cube. Joint travel is the sum of the absolute changes at all planned waypoints over all seven joints. It describes the commanded plan, not motor energy or real hardware travel.

| Approach | Full-rollout result | Mean planned joint travel, seeds 0–9 | Largest absolute angle of joints 1, 3, 5, or 7 in those plans |
| --- | --- | ---: | ---: |
| Previous local multi-start IK (`current`) | 10/10 random scenes and fixed scene passed | 100.6 rad | 360° |
| Same IK with ±180° task limits on joints 1, 3, 5, 7 | 7/10 random scenes and fixed scene passed; seeds 6, 8, 9 missed the 1 cm placement limit by about 1–2 mm | 97.9 rad | 180° |
| Task-limited IK with modest joint position feedback (`bounded`, historical default) | 10/10 random scenes and fixed scene passed | 97.9 rad | 180° |

The old planner sometimes sent joints 1 and 3 past 260° and up to the model's 360° limit even though the tool stayed in the same tabletop work area. Limiting needless winding addresses those conspicuous moves. The feedback term increases the simulated position servo's correction by 35% of its tracking error; it does not measure or correct the cube's final pose.

Other candidates were rejected after comparison:

- Stronger IK continuity regularization reduced joint travel, but an initial version could not finish seed 2. A version with stronger target-pose weighting planned the selected test scenes but failed the 1 cm placement limit in four of five full rollouts (seeds 0, 2, 7, 8). It could also choose highly wound joint branches.
- A home-posture preference reduced mean joint travel, but its stronger version failed the placement limit in seeds 3, 4, and 9. Weakening it made the benefit small and did not address winding directly.
- A phase-dependent hybrid used stronger continuity in free space and the original IK near grasp/place. It could not finish planning seed 2 and increased travel in some other scenes.

For a visual comparison, run the same seed twice:

```bash
./.venv/bin/python simulation/line_assembly.py --seed 0 --legacy-planner bounded
./.venv/bin/python simulation/line_assembly.py --seed 0 --legacy-planner current
```

This experiment selects a local IK branch and screens sampled joint transitions. It does not prove global optimality or safety for every random scene, and it is not a collision planner for real hardware.

## Single-solution reference

`--legacy-planner simple` uses the same fixed sequence of Cartesian waypoints, solves IK once per waypoint using the preceding joint pose, and rejects sampled collisions. It does not choose among multiple IK candidates or retry a wider joint range. On seed 0 it used 241 IK solves versus 723 for `bounded`; both completed 5/5 placements, but the simple route reached about 280° on a winding joint versus 180° for the bounded route. Seeds 0–9, seed 1758901330, and the fixed scene completed physical rollouts. Seed 18 stopped during planning because the single warm-started solve missed the orientation tolerance by about 0.1°. This mode is a minimal reference for comparing planning behavior, not the default.

## Random-layout failure and recovery

Seed `1758901330` exposed a discontinuity while approaching the beam splitter. With the ±180° task limit, the last 1.75 cm of planned descent changed joints 1 and 3 by about 1.3 rad each. The interpolated joint motion drove a finger pad into the table even though the endpoint pose itself had clearance. Raising the grasp point enough to plan was not a valid solution: the cube slipped during physical transfer.

The historical `bounded` planner restarts the entire plan at progressively wider winding limits when collision screening rejects every IK transition. This seed selects ±269° and passes the full physical rollout (5/5 placements, no reported table or neighbour contacts). In an additional full-rollout check of seeds 10–19, all ten passed; seeds 11 and 18 required a wider winding limit. The planner reports any relaxation so the longer arm motion is visible to the operator.
