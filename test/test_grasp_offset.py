"""Check the shared grasp offset against real Newton two-jaw contacts on CPU."""

from types import SimpleNamespace

import numpy as np
import newton
import warp as wp

from dvrk_newton.configuration import GraspConfig
from dvrk_newton.grasp import NewtonGraspManager


def test_default_offset_qualifies_distal_jaw_contacts_and_releases():
    builder = newton.ModelBuilder()
    tool = builder.add_body()
    jaws = []
    for x in (-0.002, 0.002):
        jaw = builder.add_body(xform=wp.transform((x, 0.0, 0.007), wp.quat_identity()))
        builder.body_flags[jaw] = newton.BodyFlags.KINEMATIC
        builder.add_shape_box(jaw, hx=0.001, hy=0.002, hz=0.002)
        jaws.append(jaw)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.007), wp.quat_identity()))
    shape = builder.add_shape_box(body, hx=0.0015, hy=0.0015, hz=0.0015)
    model = builder.finalize("cpu")
    state = model.state()
    collision = newton.CollisionPipeline(model)
    contacts = collision.contacts()
    collision.collide(state, contacts)
    assert contacts.rigid_contact_count.numpy()[0] >= 2
    arms = {"PSM1": SimpleNamespace(robot=SimpleNamespace(
        tool_link_index=tool,
        link_indices={"PSM1_jaw_1_link": jaws[0], "PSM1_jaw_2_link": jaws[1]},
    ))}
    objects = {"cube": SimpleNamespace(is_dynamic=True, body_index=body, shape_indices=(shape,))}
    poses, velocities = state.body_q.numpy(), state.body_qd.numpy()
    tool_pose = poses[tool].copy()
    closed = {"PSM1": SimpleNamespace(jaw_measured=0.0)}
    old = NewtonGraspManager(
        model, arms, objects, GraspConfig(contact_region_offset_m=(0.0, 0.0, -0.003))
    )
    old.step(closed, contacts, poses, velocities)
    assert not old.attachments
    manager = NewtonGraspManager(model, arms, objects)
    manager.step(closed, contacts, poses, velocities)
    assert manager.attachments["PSM1"].object_name == "cube"
    np.testing.assert_array_equal(poses[tool], tool_pose)
    manager.step({"PSM1": SimpleNamespace(jaw_measured=0.2)}, None, poses, velocities)
    assert not manager.attachments
