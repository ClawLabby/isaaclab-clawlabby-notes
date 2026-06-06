"""Standalone MJWarp elliptic zero-friction contact repro.

Run against an arbitrary MJWarp checkout by putting that checkout first on
PYTHONPATH. The unpatched checkout writes a non-finite line-search quad entry;
the patched checkout keeps the solver buffers finite.

Example:

  PYTHONPATH=/path/to/mujoco_warp uv run --extra sim --with pytest \
    python mjwarp-elliptic-friction-nan/repro_zero_friction_elliptic.py
"""

import numpy as np
import mujoco
import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import solver
from mujoco_warp._src import types


XML = """
<mujoco>
  <option cone="elliptic" solver="Newton" iterations="20" ls_iterations="20" timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 .1" condim="3" friction="1 0 0"/>
    <body pos="0 0 .095">
      <freejoint/>
      <geom type="sphere" size=".1" mass="1" condim="3" friction="1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
  _, _, m, d = test_data.fixture(
    xml=XML,
    overrides={
      "opt.jacobian": mujoco.mjtJacobian.mjJAC_DENSE,
      "opt.ls_parallel": False,
    },
  )

  mjw.step(m, d)

  # Normal MJWarp collision clamps contact friction. This deliberately models
  # an external producer, such as Newton contact conversion, writing directly
  # into Data after collision.
  d.contact.friction.fill_(types.vec5(0.0, 0.0, 0.0, 0.0, 0.0))

  mjw.make_constraint(m, d)
  ctx = solver.create_solver_context(m, d)
  solver._solve(m, d, ctx)

  nefc = int(d.nefc.numpy()[0])
  quad = ctx.quad.numpy()[0, :nefc]
  efc_force = d.efc.force.numpy()[0, :nefc]

  print(f"nefc={nefc}")
  print(f"quad_finite={bool(np.isfinite(quad).all())}")
  print(f"quad={quad.tolist()}")
  print(f"efc_force_finite={bool(np.isfinite(efc_force).all())}")
  print(f"qacc_finite={bool(np.isfinite(d.qacc.numpy()).all())}")


if __name__ == "__main__":
  main()
