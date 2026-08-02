p = params(arm_length=num(200.0, unit="mm", min=50.0, max=400.0, step=10.0))

post_solid = part.box(40, 40, 300)
arm_solid = part.box(p.arm_length, 20, 20)

post = assembly.component(post_solid, grounded=True)
arm = assembly.component(arm_solid, placement=[20, 10, 290])

hinge = assembly.joint(
    "revolute",
    assembly.connector(post, "origin",
                       offset={"position": [20, 20, 290],
                               "axis": [1, 0, 0], "angle_degrees": -90}),
    assembly.connector(arm, "origin",
                       offset={"position": [0, 10, 10],
                               "axis": [1, 0, 0], "angle_degrees": -90}),
    angle_limits_degrees=[-180.0, 180.0])

asm = assembly.assembly([post, arm], [hinge])
diag = assembly.solve(asm)

bodies = [
    assembly.body(post, density_kg_m3=7850,
                  collision=assembly.collision("box", size_mm=[40, 40, 300],
                                               offset=[20, 20, 150])),
    assembly.body(arm, density_kg_m3=2700,
                  collision=assembly.collision("box",
                                               size_mm=[p.arm_length, 20, 20],
                                               offset=[p.arm_length / 2, 10, 10])),
]

torque = assembly.actuator(hinge, kind="motor", control_nmm="0",
                           torque_limit_nmm=2000.0)
damping = assembly.joint_dynamics(hinge, damping_nmms_per_deg=0.5)

angle = assembly.observation(hinge, "position", name="hinge")
rate = assembly.observation(hinge, "velocity", name="hinge_rate")

model = assembly.mjcf(asm, bodies, actuators=[torque],
                      joint_dynamics=[damping],
                      observations=[angle, rate], label="pendulum")

task = assembly.task(model,
                     actions=[torque],
                     reward=[assembly.reward("-abs(hinge)", weight=1.0),
                             assembly.reward("1.0", weight=1.0, label="alive")],
                     episode_seconds=2.0,
                     control_hz=50,
                     label="swing")

result = {"post_solid": post_solid, "arm_solid": arm_solid,
          "post": post, "arm": arm, "hinge": hinge, "asm": asm, "diag": diag,
          "model": model, "task": task}
