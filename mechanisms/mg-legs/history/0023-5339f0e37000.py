p = params(
    hip_width=num(60, unit="mm", min=45, max=120, label="Hip width",
                  description="Distance between hip joint centers"),
    thigh_len=num(100, unit="mm", min=60, max=180, label="Thigh length"),
    calf_len=num(95, unit="mm", min=60, max=180, label="Calf length"),
    foot_len=num(70, unit="mm", min=55, max=110, label="Foot length"),
    foot_w=num(40, unit="mm", min=32, max=55, label="Foot width"),
    hip_pitch_l=num(0, unit="°", min=-60, max=90, label="L hip pitch",
                    description="Left leg swing: + forward, - back"),
    knee_pitch_l=num(0, unit="°", min=-5, max=130, label="L knee pitch",
                     description="Left knee flexion: + bends the shin back"),
    ankle_pitch_l=num(0, unit="°", min=-40, max=40, label="L ankle pitch",
                      description="Left ankle: + points the toes down"),
    hip_pitch_r=num(0, unit="°", min=-60, max=90, label="R hip pitch",
                    description="Right leg swing: + forward, - back"),
    knee_pitch_r=num(0, unit="°", min=-5, max=130, label="R knee pitch",
                     description="Right knee flexion: + bends the shin back"),
    ankle_pitch_r=num(0, unit="°", min=-40, max=40, label="R ankle pitch",
                      description="Right ankle: + points the toes down"),
    hip_roll_l=num(0, unit="°", min=-30, max=45, label="L hip roll",
                   description="Left hip roll: + tips the leg outward"),
    hip_roll_r=num(0, unit="°", min=-30, max=45, label="R hip roll",
                   description="Right hip roll: + tips the leg outward"),
    ankle_roll_l=num(0, unit="°", min=-20, max=20, label="L ankle roll",
                     description="Left ankle roll: + rolls the sole outward"),
    ankle_roll_r=num(0, unit="°", min=-20, max=20, label="R ankle roll",
                     description="Right ankle roll: + rolls the sole outward"),
)

# ---- MG90S micro servo, real dimensions (mm) ----
SV_L = 22.8      # case length
SV_W = 12.2      # case width
SV_H = 22.7      # case bottom to top deck (shaft axis direction)
SV_AX = 5.9      # output axis offset from the case front end
BOSS_R = 5.9
BOSS_H = 4.0
SPL_R = 2.35     # output spline
SPL_LEN = 6.0
FL_T = 2.5       # mounting flange thickness
FL_OVER = 4.9    # flange overhang each end (32.6 overall)
FL_DROP = 2.5    # flange sits this far below the top deck
HORN_T = 1.6
HORN_HUB = 3.6
HORN_LEN = 13.0

PL_T = 2.5       # bracket plate thickness
DXD = 9.8        # servo top-deck offset outboard of the joint centerline
# local x-slabs, + = spline / outboard side of the joint
HOLD_S = (DXD - FL_DROP, DXD)               # holder plate, spline side
HOLD_B = (DXD - SV_H - PL_T, DXD - SV_H)    # holder plate, back side
WRAP_S = (16.0, 18.5)                       # wrapper plate on the horn
WRAP_B = (-18.5, -16.0)                     # wrapper plate, back pivot
HORN_X = (14.4, 16.0)                       # horn hub/arm slab
KN_R = 9.5       # wrapper knuckle radius
HD_R = 6.0       # holder end disc radius (stays clear of the case slot)
rod_r = 2.0

foot_h = 8.0
# TWO ankle axes since B2, stacked, and the stack is what sets the height of
# everything above it.
#
# `aroll_z` is as low as the drawing allows: the roll servo's case reaches
# 5.9 mm below its own output axis and its horn arm hangs 13 mm below it, so
# 6 mm of clearance over an 8 mm sole is the floor of the floor. `ankle_z` is
# 30 mm above it, which is what leaves the pitch servo's case (mounted on the
# calf, reaching 5.9 mm below the pitch axis) clear of the roll servo's case
# top by 7.2 mm through the pitch joint's travel -- the same kind of gap the
# hip's 38 mm pitch-to-roll spacing leaves at the pelvis.
#
# THE MACHINE IS 20 mm TALLER FOR IT, and that is a real cost paid in the
# open rather than hidden: a taller centre of mass means a smaller omega0 =
# sqrt(g/h), which means the same impulse throws the capture point further.
# The alternative -- shortening `calf_len` by 20 mm so the hip stays where it
# was -- was rejected because it would make the leg's REACH quietly wrong in
# every place this file reasons about a step, and reach is what bounds the
# shoves the task declares. If the height turns out to hurt, `calf_len` is
# the one parameter to move, deliberately and with the reach recomputed.
aroll_z = foot_h + 6.0
ankle_z = aroll_z + 30.0
knee_z = ankle_z + p.calf_len
hip_z = knee_z + p.thigh_len
roll_z = hip_z + 38.0
pelvis_z = roll_z + 27.0
hx = p.hip_width / 2.0


def xslab(xc, s, ab):
    """World x of the lower face + extent for a local slab, mirrored by s."""
    a, b = ab
    x0 = xc + a if s > 0 else xc - b
    return x0, b - a


def shaped_plate(x0, z0, z1, w0, w1, wm, mf, slope=0.0):
    """One bracket plate: organic tapered outline, extruded PL_T along X.
    w0/w1 = half-widths at bottom/top, wm = bulge at fraction mf.
    slope leans the plate in X per unit Z."""
    zm = z0 + mf * (z1 - z0)
    xm = x0 + slope * (zm - z0)
    x1 = x0 + slope * (z1 - z0)
    right = part.bspline([(x0, w0, z0), (xm, wm, zm), (x1, w1, z1)])
    top = part.line((x1, w1, z1), (x1, -w1, z1))
    left = part.bspline([(x1, -w1, z1), (xm, -wm, zm), (x0, -w0, z0)])
    bottom = part.line((x0, -w0, z0), (x0, w0, z0))
    outline = part.wire([right, top, left, bottom], closed=True)
    return part.extrude(part.face(outline), (PL_T, 0.0, 0.0))


def disc_x(xc, s, ab, z, r):
    x0, ln = xslab(xc, s, ab)
    return part.cylinder(r, ln, origin=(x0, 0.0, z), direction=(1.0, 0.0, 0.0))


def rod_between(xc, s, z, a, b):
    x0 = xc + a if s > 0 else xc - b
    return part.cylinder(rod_r, b - a, origin=(x0, 0.0, z),
                         direction=(1.0, 0.0, 0.0))


def slot_cutter(x, span, slot_w, z1, z2):
    """Rounded lightening window punched through both plates along X."""
    cx = x - span / 2.0 - 1.0
    reach = span + 2.0
    return part.fuse([
        part.box(reach, slot_w, z2 - z1, origin=(cx, -slot_w / 2.0, z1)),
        part.cylinder(slot_w / 2.0, reach,
                      origin=(cx, 0.0, z1), direction=(1.0, 0.0, 0.0)),
        part.cylinder(slot_w / 2.0, reach,
                      origin=(cx, 0.0, z2), direction=(1.0, 0.0, 0.0)),
    ])


# ---- servo bodies ----
def servo_x(xc, s, jz):
    """MG90S with its output axis along X at height jz, spline outboard,
    case body extending upward (long side above the axis)."""
    x0, _ = xslab(xc, s, (DXD - SV_H, DXD))
    case = part.box(SV_H, SV_W, SV_L, origin=(x0, -SV_W / 2.0, jz - SV_AX))
    fx, _ = xslab(xc, s, (DXD - FL_DROP - FL_T, DXD - FL_DROP))
    flange = part.box(FL_T, SV_W, SV_L + 2.0 * FL_OVER,
                      origin=(fx, -SV_W / 2.0, jz - SV_AX - FL_OVER))
    dx = xc + s * (DXD - 1.0)
    boss = part.cylinder(BOSS_R, BOSS_H + 1.0, origin=(dx, 0.0, jz),
                         direction=(s, 0.0, 0.0))
    spl = part.cylinder(SPL_R, SPL_LEN + 1.0, origin=(dx, 0.0, jz),
                        direction=(s, 0.0, 0.0))
    return part.fuse([case, flange, boss, spl])


def servo_y(xc, jz):
    """MG90S with its output axis along Y (hip roll), spline forward."""
    case = part.box(SV_W, SV_H, SV_L,
                    origin=(xc - SV_W / 2.0, DXD - SV_H, jz - SV_AX))
    flange = part.box(SV_W, FL_T, SV_L + 2.0 * FL_OVER,
                      origin=(xc - SV_W / 2.0, DXD - FL_DROP - FL_T,
                              jz - SV_AX - FL_OVER))
    boss = part.cylinder(BOSS_R, BOSS_H + 1.0, origin=(xc, DXD - 1.0, jz),
                         direction=(0.0, 1.0, 0.0))
    spl = part.cylinder(SPL_R, SPL_LEN + 1.0, origin=(xc, DXD - 1.0, jz),
                        direction=(0.0, 1.0, 0.0))
    return part.fuse([case, flange, boss, spl])


def horn_x(xc, s, jz):
    """Servo horn (hub + single arm pointing down), fused into the driven
    link and overlapping its wrapper plate on the spline side."""
    x0, ln = xslab(xc, s, (HORN_X[0], WRAP_S[0] + 0.5))
    return [part.cylinder(HORN_HUB, ln, origin=(x0, 0.0, jz),
                          direction=(1.0, 0.0, 0.0)),
            part.box(ln, 6.0, HORN_LEN, origin=(x0, -3.0, jz - HORN_LEN)),
            part.cylinder(3.0, ln, origin=(x0, 0.0, jz - HORN_LEN),
                          direction=(1.0, 0.0, 0.0))]


def stub_x(xc, s, jz):
    """Back-side pivot stub from the wrapper plate to the servo bottom."""
    x0, ln = xslab(xc, s, (WRAP_B[1] - 1.0, HOLD_B[1]))
    return part.cylinder(2.4, ln, origin=(x0, 0.0, jz),
                         direction=(1.0, 0.0, 0.0))


def horn_y(xc, jz):
    t = HORN_T + 0.5
    return [part.cylinder(HORN_HUB, t, origin=(xc, HORN_X[0], jz),
                          direction=(0.0, 1.0, 0.0)),
            part.box(6.0, t, HORN_LEN,
                     origin=(xc - 3.0, HORN_X[0], jz - HORN_LEN)),
            part.cylinder(3.0, t, origin=(xc, HORN_X[0], jz - HORN_LEN),
                          direction=(0.0, 1.0, 0.0))]


def stub_y(xc, jz):
    return part.cylinder(2.4, HOLD_B[1] - WRAP_B[1] + 1.0,
                         origin=(xc, WRAP_B[1] - 1.0, jz),
                         direction=(0.0, 1.0, 0.0))


def slot_x(xc, s, jz):
    """Rectangular pass-through for the servo case top in the spline-side
    holder plate (cut generously to cover the plate's lean)."""
    x0, ln = xslab(xc, s, (6.8, 12.4))
    return part.box(ln, SV_W + 0.4, SV_L + 0.4,
                    origin=(x0, -(SV_W + 0.4) / 2.0, jz - SV_AX - 0.2))


def slot_y(xc, jz):
    return part.box(SV_W + 0.4, 3.5, SV_L + 0.4,
                    origin=(xc - (SV_W + 0.4) / 2.0, 6.8, jz - SV_AX - 0.2))


# ---- links ----
def make_thigh(xc, s):
    """Holds the knee servo between its plates at the bottom; wraps the hip
    link's pitch servo at the top (horn one side, pivot stub the other)."""
    L = p.thigh_len
    sp = shaped_plate(xslab(xc, s, HOLD_S)[0], knee_z, hip_z,
                      8.0, 9.5, 11.0, 0.68,
                      slope=s * (WRAP_S[0] - HOLD_S[0]) / L)
    bp = shaped_plate(xslab(xc, s, HOLD_B)[0], knee_z, hip_z,
                      8.0, 9.5, 11.0, 0.68,
                      slope=s * (WRAP_B[0] - HOLD_B[0]) / L)
    discs = [disc_x(xc, s, HOLD_S, knee_z, HD_R),
             disc_x(xc, s, HOLD_B, knee_z, HD_R),
             disc_x(xc, s, WRAP_S, hip_z, KN_R),
             disc_x(xc, s, WRAP_B, hip_z, KN_R)]

    def rodz(z):
        f = (z - knee_z) / L
        a = HOLD_B[0] + f * (WRAP_B[0] - HOLD_B[0])
        b = HOLD_S[1] + f * (WRAP_S[1] - HOLD_S[1])
        return rod_between(xc, s, z, a, b)

    rods = [rodz(knee_z + 20.0), rodz(hip_z - 20.0)]
    thigh = part.fuse([sp, bp] + discs + rods
                      + horn_x(xc, s, hip_z) + [stub_x(xc, s, hip_z)])
    cutters = [slot_x(xc, s, knee_z),
               slot_cutter(xc, 44.0, 9.0, knee_z + 28.0, hip_z - 28.0)]
    return part.cut(thigh, cutters, label="thigh")


def make_calf(xc, s):
    """Holds the ankle servo at the bottom; wraps the knee servo (mounted
    in the thigh) at the top."""
    L = p.calf_len
    sp = shaped_plate(xslab(xc, s, HOLD_S)[0], ankle_z, knee_z,
                      8.0, 9.5, 10.5, 0.7,
                      slope=s * (WRAP_S[0] - HOLD_S[0]) / L)
    bp = shaped_plate(xslab(xc, s, HOLD_B)[0], ankle_z, knee_z,
                      8.0, 9.5, 10.5, 0.7,
                      slope=s * (WRAP_B[0] - HOLD_B[0]) / L)
    discs = [disc_x(xc, s, HOLD_S, ankle_z, HD_R),
             disc_x(xc, s, HOLD_B, ankle_z, HD_R),
             disc_x(xc, s, WRAP_S, knee_z, KN_R),
             disc_x(xc, s, WRAP_B, knee_z, KN_R)]

    def rodz(z):
        f = (z - ankle_z) / L
        a = HOLD_B[0] + f * (WRAP_B[0] - HOLD_B[0])
        b = HOLD_S[1] + f * (WRAP_S[1] - HOLD_S[1])
        return rod_between(xc, s, z, a, b)

    rods = [rodz(ankle_z + 20.0), rodz(knee_z - 20.0)]
    calf = part.fuse([sp, bp] + discs + rods
                     + horn_x(xc, s, knee_z) + [stub_x(xc, s, knee_z)])
    cutters = [slot_x(xc, s, ankle_z),
               slot_cutter(xc, 44.0, 9.0, ankle_z + 28.0, knee_z - 28.0)]
    return part.cut(calf, cutters, label="calf")


def make_hip(xc, s):
    """Intermediate link: holds the hip-pitch servo between vertical
    plates, and wraps the pelvis-mounted roll servo above (horn on the
    front, pivot stub on the back)."""
    pieces = []
    for ab in (HOLD_S, HOLD_B):
        x0, _ = xslab(xc, s, ab)
        pieces.append(part.box(PL_T, 17.0, 21.0,
                               origin=(x0, -8.5, hip_z)))
        pieces.append(disc_x(xc, s, ab, hip_z, HD_R))
    cx0, cln = xslab(xc, s, (HOLD_B[0], HOLD_S[1]))
    pieces.append(part.box(cln, 37.0, 6.0,
                           origin=(cx0, -18.5, hip_z + 20.0)))
    for a, b in (WRAP_S, WRAP_B):
        pieces.append(part.box(16.0, PL_T, roll_z - hip_z - 24.0,
                               origin=(xc - 8.0, a, hip_z + 24.0)))
        pieces.append(part.cylinder(KN_R, PL_T, origin=(xc, a, roll_z),
                                    direction=(0.0, 1.0, 0.0)))
    pieces += horn_y(xc, roll_z) + [stub_y(xc, roll_z)]
    link = part.fuse(pieces)
    return part.cut(link, slot_x(xc, s, hip_z), label="hip")


# ---- foot with passive toe hinge ----
ball_r = foot_h / 2.0
ball_pin_r = foot_h * 0.28
toe_clr = 0.5


def footprint(x):
    hw = p.foot_w / 2.0
    L = p.foot_len
    y0 = -0.35 * L
    pts = [
        (x, y0, 0.0),
        (x + 0.80 * hw, y0 + 0.10 * L, 0.0),
        (x + 0.72 * hw, y0 + 0.40 * L, 0.0),
        (x + hw, y0 + 0.68 * L, 0.0),
        (x + 0.80 * hw, y0 + 0.92 * L, 0.0),
        (x, y0 + L, 0.0),
        (x - 0.80 * hw, y0 + 0.92 * L, 0.0),
        (x - hw, y0 + 0.68 * L, 0.0),
        (x - 0.72 * hw, y0 + 0.40 * L, 0.0),
        (x - 0.80 * hw, y0 + 0.10 * L, 0.0),
    ]
    outline = part.wire([part.bspline(pts, periodic=True)], closed=True)
    return part.extrude(part.face(outline), (0.0, 0.0, foot_h))


def make_ankle_bracket(xc, s):
    """Between the two ankle axes, and it is `make_hip` upside down.

    That link already does exactly this job at the pelvis: hold one servo
    between two plates and wrap the perpendicular one on the far side. Here
    the wrapped servo is the calf's PITCH servo above, and the held one is
    the new ROLL servo below, whose horn drives the foot.

    The two plate sets never meet, which is what makes the drawing work in
    30 mm: the pitch wrap is two X-slabs at |x| 16-18.5 mm and the roll
    servo is 6.1 mm either side of the centreline, so the wrap can reach
    down past the servo without touching it. Only the web tying the two
    holder plates together has to clear the roll case, and it sits above it.
    """

    pieces = []
    for a, b in (HOLD_S, HOLD_B):
        pieces.append(part.box(16.0, PL_T, ankle_z - aroll_z - 8.0,
                               origin=(xc - 8.0, a, aroll_z)))
        pieces.append(part.cylinder(HD_R, PL_T, origin=(xc, a, aroll_z),
                                    direction=(0.0, 1.0, 0.0)))
    # The one plate that makes this ONE solid: it has to reach outboard far
    # enough to meet the pitch wrap's X-slabs at |x| 16-18.5 AND back far
    # enough in Y to meet the roll holder's back plate, because those two
    # plate sets share no coordinate otherwise. Without it the fuse comes
    # back as a compound of three -- the holders, and each wrap slab with its
    # own knuckle -- which is exactly what the engine refused first time.
    # It sits above the roll servo's case top, which is what the 30 mm axis
    # spacing bought.
    pieces.append(part.box(WRAP_S[1] - WRAP_B[0], HOLD_S[1] - HOLD_B[0], 4.0,
                           origin=(xc + WRAP_B[0], HOLD_B[0],
                                   ankle_z - 12.0)))
    for ab in (WRAP_S, WRAP_B):
        x0, ln = xslab(xc, s, ab)
        pieces.append(part.box(ln, 17.0, ankle_z - aroll_z - 12.0,
                               origin=(x0, -8.5, aroll_z + 12.0)))
        pieces.append(disc_x(xc, s, ab, ankle_z, KN_R))
    pieces += horn_x(xc, s, ankle_z) + [stub_x(xc, s, ankle_z)]
    link = part.fuse(pieces)
    return part.cut(link, slot_y(xc, aroll_z), label="ankle_bracket")


def make_foot(xc):
    hw = p.foot_w / 2.0
    heel = 0.35 * p.foot_len
    ball_y = 0.35 * p.foot_len
    tong_hw = 0.26 * p.foot_w
    body = part.common([footprint(xc),
                        part.box(p.foot_w + 4.0, ball_y + heel + 1.0,
                                 foot_h + 2.0,
                                 origin=(xc - hw - 2.0, -heel - 1.0, -1.0))])
    notch = part.box(2.0 * (tong_hw + toe_clr),
                     ball_r + toe_clr + 1.0, foot_h + 2.0,
                     origin=(xc - tong_hw - toe_clr,
                             ball_y - ball_r - toe_clr, -1.0))
    body = part.cut(body, notch)
    span = 0.98 * hw
    lug_l = part.cylinder(ball_r, span - tong_hw - toe_clr,
                          origin=(xc - span, ball_y, ball_r),
                          direction=(1.0, 0.0, 0.0))
    lug_r = part.cylinder(ball_r, span - tong_hw - toe_clr,
                          origin=(xc + tong_hw + toe_clr, ball_y, ball_r),
                          direction=(1.0, 0.0, 0.0))
    ball_pin = part.cylinder(ball_pin_r, 2.0 * span,
                             origin=(xc - span, ball_y, ball_r),
                             direction=(1.0, 0.0, 0.0))
    # Since B2 the foot hangs off the ROLL axis rather than the pitch one, so
    # what it carries is the roll wrap: two plates in Y standing off the sole,
    # their knuckles at the roll axis, the horn on the front and the pivot
    # stub behind. That is the same wrap the hip link puts on the pelvis's
    # roll servo, and it is deliberately not mirrored -- a Y-axis servo has a
    # front and a back rather than a left and a right.
    pad = part.box(16.0, WRAP_S[1] - WRAP_B[0], 4.0,
                   origin=(xc - 8.0, WRAP_B[0], foot_h - 1.0))
    plates = [part.box(16.0, PL_T, aroll_z - foot_h + 1.0,
                       origin=(xc - 8.0, a, foot_h - 1.0))
              for a, b in (WRAP_S, WRAP_B)]
    knuckles = [part.cylinder(KN_R, PL_T, origin=(xc, a, aroll_z),
                              direction=(0.0, 1.0, 0.0))
                for a, b in (WRAP_S, WRAP_B)]
    return part.fuse([body, lug_l, lug_r, ball_pin, pad]
                     + plates + knuckles
                     + horn_y(xc, aroll_z) + [stub_y(xc, aroll_z)],
                     label="foot")


def make_toe(xc):
    hw = p.foot_w / 2.0
    ball_y = 0.35 * p.foot_len
    ytip = 0.65 * p.foot_len
    tong_hw = 0.26 * p.foot_w
    body = part.common([footprint(xc),
                        part.box(p.foot_w + 4.0,
                                 ytip - ball_y - ball_r - toe_clr + 1.0,
                                 foot_h + 2.0,
                                 origin=(xc - hw - 2.0,
                                         ball_y + ball_r + toe_clr, -1.0))])
    tongue = part.fuse([
        part.cylinder(ball_r, 2.0 * tong_hw,
                      origin=(xc - tong_hw, ball_y, ball_r),
                      direction=(1.0, 0.0, 0.0)),
        part.box(2.0 * tong_hw, ball_r + toe_clr + 2.0, foot_h,
                 origin=(xc - tong_hw, ball_y, 0.0)),
    ])
    toe = part.fuse([body, tongue])
    bore = part.cylinder(ball_pin_r + toe_clr, 2.0 * tong_hw + 2.0,
                         origin=(xc - tong_hw - 1.0, ball_y, ball_r),
                         direction=(1.0, 0.0, 0.0))
    return part.cut(toe, bore, label="toe")


# ---- pelvis: crossbar, sacrum, spine, wings, roll-servo holders ----
crossbar = part.cylinder(7.0, p.hip_width + 20.0,
                         origin=(-hx - 10.0, 0.0, pelvis_z),
                         direction=(1.0, 0.0, 0.0))
sacrum_w = 0.6 * p.hip_width
sacrum = part.box(sacrum_w, 16.0, 14.0,
                  origin=(-sacrum_w / 2.0, -8.0, pelvis_z))
spine = part.cylinder(6.0, 22.0, origin=(0.0, 0.0, pelvis_z))
wing_l = part.cone(9.0, 4.5, 20.0, origin=(-hx, 0.0, pelvis_z),
                   direction=(-0.35, 0.0, 1.0))
wing_r = part.cone(9.0, 4.5, 20.0, origin=(hx, 0.0, pelvis_z),
                   direction=(0.35, 0.0, 1.0))
holders = []
for xc in (-hx, hx):
    for a, b in (HOLD_S, HOLD_B):
        holders.append(part.box(16.0, PL_T, pelvis_z - roll_z,
                                origin=(xc - 8.0, a, roll_z)))
        holders.append(part.cylinder(HD_R, PL_T, origin=(xc, a, roll_z),
                                     direction=(0.0, 1.0, 0.0)))
    holders.append(part.box(16.0, HOLD_S[1] - HOLD_B[0], 8.0,
                            origin=(xc - 8.0, HOLD_B[0], pelvis_z - 4.0)))
pelvis = part.fuse([crossbar, sacrum, spine, wing_l, wing_r] + holders)
pelvis = part.cut(pelvis, [slot_y(-hx, roll_z), slot_y(hx, roll_z)],
                  label="pelvis")


floor_solid = part.box(1200.0, 1200.0, 40.0, origin=(-600.0, -600.0, -40.0))


# ---------------------------------------------------------------------------
# Posing, and why the JOINT FRAMES are posed rather than the components.
# ---------------------------------------------------------------------------
#
# The sliders still rotate the SOLIDS, exactly as they did before there was an
# assembly here. What is new is that each joint's connector frame is carried
# through the SAME rotation sequence, so a joint is anchored where its geometry
# actually is rather than where the neutral drawing left it.
#
# The obvious alternative -- neutral solids, and the pose in
# `assembly.component(placement=...)` -- was built for the sibling `legs`
# project and MEASURED, and it does not survive the native solver. A component
# placement is only a starting point: an island the joints never reach from
# ground has six free degrees of freedom, and the solver answers such a system
# with its own member of the solution family. On that machine it zeroed all
# eight joints and displaced the whole biped by (90, 18, 58) mm and 40 degrees.
# Nothing is wrong with that answer -- every joint is satisfied -- but the pose
# the sliders asked for is gone, and a floating base is exactly the case where
# it happens.
#
# So the rule this script follows is: **the two connectors of a joint carry the
# identical WORLD frame, and that frame is posed.** The residual is then zero
# at whatever the sliders say, the solver has nothing to collapse, and the
# exported model is the machine that is drawn.
#
# xscript has no `import`, so sin and cos are series here.

PI = 3.141592653589793
SQRT_HALF = 0.7071067811865476


def _sin(a):
    t = a
    while t > PI:
        t = t - 2.0 * PI
    while t < -PI:
        t = t + 2.0 * PI
    if t > 0.5 * PI:
        t = PI - t
    elif t < -0.5 * PI:
        t = -PI - t
    x2 = t * t
    return t * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (
        -1.0 / 5040.0 + x2 * (1.0 / 362880.0 + x2 * (
            -1.0 / 39916800.0 + x2 * (
                1.0 / 6227020800.0 - x2 / 1307674368000.0)))))))


def _cos(a):
    return _sin(a + 0.5 * PI)


def _q_axis(axis, degrees):
    """Unit quaternion [x, y, z, w] for a turn about a unit axis."""
    half = 0.5 * degrees * PI / 180.0
    s = _sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, _cos(half)]


def _q_mul(a, b):
    """a . b -- b's rotation happens first."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def _q_rot(q, v):
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx)]


NEUTRAL = ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
XAXIS = (1.0, 0.0, 0.0)
YAXIS = (0.0, 1.0, 0.0)


def turn(place, axis, degrees, pivot):
    """One more rotation about a world axis through a pivot, composed onto a
    placement: q' = qr . q and t' = R(qr) . (t - pivot) + pivot. This is
    `part.transform(..., pivot=...)` performed on numbers instead of on a
    solid, which is the whole point -- one list drives both."""
    if degrees == 0.0:
        return place
    q, t = place
    qr = _q_axis(axis, degrees)
    moved = _q_rot(qr, [t[0] - pivot[0], t[1] - pivot[1], t[2] - pivot[2]])
    return (_q_mul(qr, q),
            [moved[0] + pivot[0], moved[1] + pivot[1], moved[2] + pivot[2]])


def leg_turns(x, hip_a, knee_a, ankle_a, roll_a, aroll_a):
    """The rotation sequence each part of one leg gets, innermost first.

    These are the turns `pose_leg` used to apply to solids, written once so
    that the same list drives a solid AND that solid's joint anchors. Five
    of them since B2: the ankle roll is the fifth, and it sits between the
    ankle bracket and the foot, so everything from the foot down carries it
    and everything above the roll axis does not."""
    s = 1.0 if x > 0 else -1.0
    roll = (YAXIS, -s * roll_a, (x, 0.0, roll_z))       # + tips leg outward
    hip = (XAXIS, hip_a, (0.0, 0.0, hip_z))             # + swings it forward
    knee = (XAXIS, -knee_a, (0.0, 0.0, knee_z))         # + bends shin back
    ankle = (XAXIS, -ankle_a, (0.0, 0.0, ankle_z))      # + points toes down
    aroll = (YAXIS, -s * aroll_a, (x, 0.0, aroll_z))    # + rolls sole out
    return ([roll],
            [hip, roll],
            [knee, hip, roll],
            [ankle, knee, hip, roll],
            [aroll, ankle, knee, hip, roll],
            [aroll, ankle, knee, hip, roll])


def posed_solid(shape, turns):
    for entry in turns:
        shape = part.transform(shape, rotation_axis=entry[0],
                               rotation_degrees=entry[1], pivot=entry[2])
    return shape


def posed_place(turns):
    place = NEUTRAL
    for entry in turns:
        place = turn(place, entry[0], entry[1], entry[2])
    return place


def at(place, point):
    """One point of the neutral drawing, where the pose actually put it."""
    q, t = place
    r = _q_rot(q, [point[0], point[1], point[2]])
    return [r[0] + t[0], r[1] + t[1], r[2] + t[2]]


def minus(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


# ---------------------------------------------------------------------------
# Which link each servo belongs to.
# ---------------------------------------------------------------------------
#
# `pose_leg` already knew this -- it carried each servo through the rotations
# of the link it is bolted into, and put the matching horn in the link that
# servo drives.
#
# What is new is that a servo is a separate COMPONENT, welded to its carrier,
# rather than fused into it. That costs eight bodies and eight fixed joints,
# and it buys the only thing `assembly.body` cannot otherwise express: a
# SECOND DENSITY. Mass is `density_kg_m3` times the solid's own volume and
# there is no mass override, so a servo fused into a printed bracket would
# have to be made of the same stuff as the bracket.
#
# It is not a rounding error. Eight MG90S at 13.4 g are about two fifths of
# what this machine weighs, sitting exactly ON the joints, where they do the
# most to the inertia the servo has to fight. Printing them out of PLA would
# lose a third of that and quietly flatter every torque in feasibility.py.
#
#   roll servo    -> pelvis         (horn drives the hip link)
#   pitch servo   -> hip            (horn drives the thigh)
#   knee servo    -> thigh          (horn drives the calf)
#   ankle servo   -> calf           (horn drives the ankle bracket)
#   a-roll servo  -> ankle bracket  (horn drives the foot)
#
# Ten servos since B2, not eight. Two more MG90S is +26.8 g on 263 g and the
# two of them sit at the ankles -- the lowest place on the machine a servo
# can go, which helps the standing centre of mass, and the far end of the
# swing chain, which hurts a step. Both are true and the second is the price
# of the first.


def leg_links(xc, s):
    """The six links of one leg, neutral, WITHOUT their servos."""
    return (make_hip(xc, s), make_thigh(xc, s), make_calf(xc, s),
            make_ankle_bracket(xc, s), make_foot(xc), make_toe(xc))


def servo_centre_x(xc, s, jz):
    """Middle of an MG90S case whose output axis runs along X at `jz`."""
    x0 = xc + (DXD - SV_H) if s > 0 else xc - DXD
    return [x0 + 0.5 * SV_H, 0.0, jz - SV_AX + 0.5 * SV_L]


def servo_centre_y(xc, jz):
    """Middle of the roll servo, whose output axis runs along Y."""
    return [xc, DXD - 0.5 * SV_H, jz - SV_AX + 0.5 * SV_L]


# ---------------------------------------------------------------------------
# Component frames, and why each part gets its own.
# ---------------------------------------------------------------------------
#
# Every solid above is drawn in ONE frame -- the standing figure's -- which is
# the natural way to draw a figure and the wrong way to hand it to an assembly.
# If every component sits at the identity then every body's frame coincides
# with its parent's, so every relative placement in the exported model is the
# identity, and the ONE thing MJCF's field-drift check divides by is the
# largest entry of a field whose every entry should be zero. On `legs` it
# measured 1.0 and refused the export, correctly.
#
# So each part is moved onto its OWN origin and the component carries that
# origin as a pure translation. The origin is each limb's MIDDLE rather than
# its proximal joint, which is the second half of the same lesson: `jnt_pos`
# is a joint's position in its CHILD's frame, so a frame at the proximal joint
# moves the degeneracy to that field instead of removing it.


def local(shape, origin):
    return part.transform(shape, translation=[-origin[0], -origin[1],
                                              -origin[2]])


# A revolute turns about its connectors' local +Z, so each frame carries the
# quarter turn that points +Z along the axis the drawing already uses: ROLL_Z
# sends +Z to world +Y (the pelvis roll pins) and PITCH_Z sends it to world +X
# (every pitch axle). Written as exact quaternions rather than through the
# series, because these two are the same at every pose.
ROLL_Z = [-SQRT_HALF, 0.0, 0.0, SQRT_HALF]
PITCH_Z = [0.0, SQRT_HALF, 0.0, SQRT_HALF]

turns_l = leg_turns(-hx, p.hip_pitch_l, p.knee_pitch_l,
                    p.ankle_pitch_l, p.hip_roll_l, p.ankle_roll_l)
turns_r = leg_turns(hx, p.hip_pitch_r, p.knee_pitch_r,
                    p.ankle_pitch_r, p.hip_roll_r, p.ankle_roll_r)
place_l = [posed_place(entry) for entry in turns_l]
place_r = [posed_place(entry) for entry in turns_r]

hip_n_l, thigh_n_l, calf_n_l, brk_n_l, foot_n_l, toe_n_l = leg_links(-hx, -1.0)
hip_n_r, thigh_n_r, calf_n_r, brk_n_r, foot_n_r, toe_n_r = leg_links(hx, 1.0)

# Each part's own origin: the middle of its own limb, where the pose put it.
ball_y = 0.35 * p.foot_len
hip_mid = 0.5 * (roll_z + hip_z)
thigh_mid = 0.5 * (hip_z + knee_z)
calf_mid = 0.5 * (knee_z + ankle_z)
brk_mid = 0.5 * (aroll_z + ankle_z)
foot_mid_y = 0.5 * ball_y
foot_mid_z = 0.5 * (aroll_z + ball_r)

pelvis_o = [0.0, 0.0, pelvis_z]
hip_o_l = at(place_l[0], (-hx, 0.0, hip_mid))
hip_o_r = at(place_r[0], (hx, 0.0, hip_mid))
thigh_o_l = at(place_l[1], (-hx, 0.0, thigh_mid))
thigh_o_r = at(place_r[1], (hx, 0.0, thigh_mid))
calf_o_l = at(place_l[2], (-hx, 0.0, calf_mid))
calf_o_r = at(place_r[2], (hx, 0.0, calf_mid))
brk_o_l = at(place_l[3], (-hx, 0.0, brk_mid))
brk_o_r = at(place_r[3], (hx, 0.0, brk_mid))
foot_o_l = at(place_l[4], (-hx, foot_mid_y, foot_mid_z))
foot_o_r = at(place_r[4], (hx, foot_mid_y, foot_mid_z))
toe_o_l = at(place_l[5], (-hx, ball_y, ball_r))
toe_o_r = at(place_r[5], (hx, ball_y, ball_r))

# Each servo rides the link it is bolted into, so it takes that link's turns:
# the roll servos sit in the pelvis and never move, the pitch servo rides the
# hip link, the knee servo the thigh, the ankle servo the calf.
sv_roll_o_l = servo_centre_y(-hx, roll_z)
sv_roll_o_r = servo_centre_y(hx, roll_z)
sv_pitch_o_l = at(place_l[0], servo_centre_x(-hx, -1.0, hip_z))
sv_pitch_o_r = at(place_r[0], servo_centre_x(hx, 1.0, hip_z))
sv_knee_o_l = at(place_l[1], servo_centre_x(-hx, -1.0, knee_z))
sv_knee_o_r = at(place_r[1], servo_centre_x(hx, 1.0, knee_z))
sv_ankle_o_l = at(place_l[2], servo_centre_x(-hx, -1.0, ankle_z))
sv_ankle_o_r = at(place_r[2], servo_centre_x(hx, 1.0, ankle_z))
# ...and the ankle-roll servo rides the bracket, which is place[3].
sv_aroll_o_l = at(place_l[3], servo_centre_y(-hx, aroll_z))
sv_aroll_o_r = at(place_r[3], servo_centre_y(hx, aroll_z))

sv_roll_l = local(servo_y(-hx, roll_z), sv_roll_o_l)
sv_roll_r = local(servo_y(hx, roll_z), sv_roll_o_r)
sv_pitch_l = local(posed_solid(servo_x(-hx, -1.0, hip_z), turns_l[0]),
                   sv_pitch_o_l)
sv_pitch_r = local(posed_solid(servo_x(hx, 1.0, hip_z), turns_r[0]),
                   sv_pitch_o_r)
sv_knee_l = local(posed_solid(servo_x(-hx, -1.0, knee_z), turns_l[1]),
                  sv_knee_o_l)
sv_knee_r = local(posed_solid(servo_x(hx, 1.0, knee_z), turns_r[1]),
                  sv_knee_o_r)
sv_ankle_l = local(posed_solid(servo_x(-hx, -1.0, ankle_z), turns_l[2]),
                   sv_ankle_o_l)
sv_ankle_r = local(posed_solid(servo_x(hx, 1.0, ankle_z), turns_r[2]),
                   sv_ankle_o_r)
sv_aroll_l = local(posed_solid(servo_y(-hx, aroll_z), turns_l[3]),
                   sv_aroll_o_l)
sv_aroll_r = local(posed_solid(servo_y(hx, aroll_z), turns_r[3]),
                   sv_aroll_o_r)

pelvis_solid = local(pelvis, pelvis_o)
hip_l = local(posed_solid(hip_n_l, turns_l[0]), hip_o_l)
hip_r = local(posed_solid(hip_n_r, turns_r[0]), hip_o_r)
thigh_l = local(posed_solid(thigh_n_l, turns_l[1]), thigh_o_l)
thigh_r = local(posed_solid(thigh_n_r, turns_r[1]), thigh_o_r)
calf_l = local(posed_solid(calf_n_l, turns_l[2]), calf_o_l)
calf_r = local(posed_solid(calf_n_r, turns_r[2]), calf_o_r)
brk_l = local(posed_solid(brk_n_l, turns_l[3]), brk_o_l)
brk_r = local(posed_solid(brk_n_r, turns_r[3]), brk_o_r)
foot_l = local(posed_solid(foot_n_l, turns_l[4]), foot_o_l)
foot_r = local(posed_solid(foot_n_r, turns_r[4]), foot_o_r)
toe_l = local(posed_solid(toe_n_l, turns_l[5]), toe_o_l)
toe_r = local(posed_solid(toe_n_r, turns_r[5]), toe_o_r)


# ---------------------------------------------------------------------------
# The assembly: twelve components, ten joints, and NO joint to the floor.
# ---------------------------------------------------------------------------
#
# That last absence is the point of the machine. The floor is grounded and no
# joint reaches the pelvis from it, so the pelvis -- FIRST in the components
# list, which is what decides it -- gets a free joint and the biped is a
# floating base that falls over.

ground = assembly.component(floor_solid, grounded=True, label="ground")
pelvis_c = assembly.component(pelvis_solid, placement={"position": pelvis_o},
                              label="pelvis")
hip_l_c = assembly.component(hip_l, placement={"position": hip_o_l},
                             label="hip_l")
hip_r_c = assembly.component(hip_r, placement={"position": hip_o_r},
                             label="hip_r")
thigh_l_c = assembly.component(thigh_l, placement={"position": thigh_o_l},
                               label="thigh_l")
thigh_r_c = assembly.component(thigh_r, placement={"position": thigh_o_r},
                               label="thigh_r")
calf_l_c = assembly.component(calf_l, placement={"position": calf_o_l},
                              label="calf_l")
calf_r_c = assembly.component(calf_r, placement={"position": calf_o_r},
                              label="calf_r")
brk_l_c = assembly.component(brk_l, placement={"position": brk_o_l},
                             label="ankle_bracket_l")
brk_r_c = assembly.component(brk_r, placement={"position": brk_o_r},
                             label="ankle_bracket_r")
foot_l_c = assembly.component(foot_l, placement={"position": foot_o_l},
                              label="foot_l")
foot_r_c = assembly.component(foot_r, placement={"position": foot_o_r},
                              label="foot_r")
toe_l_c = assembly.component(toe_l, placement={"position": toe_o_l},
                             label="toe_l")
toe_r_c = assembly.component(toe_r, placement={"position": toe_o_r},
                             label="toe_r")
sv_roll_l_c = assembly.component(sv_roll_l,
                                 placement={"position": sv_roll_o_l},
                                 label="sv_roll_l")
sv_roll_r_c = assembly.component(sv_roll_r,
                                 placement={"position": sv_roll_o_r},
                                 label="sv_roll_r")
sv_pitch_l_c = assembly.component(sv_pitch_l,
                                  placement={"position": sv_pitch_o_l},
                                  label="sv_pitch_l")
sv_pitch_r_c = assembly.component(sv_pitch_r,
                                  placement={"position": sv_pitch_o_r},
                                  label="sv_pitch_r")
sv_knee_l_c = assembly.component(sv_knee_l,
                                 placement={"position": sv_knee_o_l},
                                 label="sv_knee_l")
sv_knee_r_c = assembly.component(sv_knee_r,
                                 placement={"position": sv_knee_o_r},
                                 label="sv_knee_r")
sv_ankle_l_c = assembly.component(sv_ankle_l,
                                  placement={"position": sv_ankle_o_l},
                                  label="sv_ankle_l")
sv_ankle_r_c = assembly.component(sv_ankle_r,
                                  placement={"position": sv_ankle_o_r},
                                  label="sv_ankle_r")
sv_aroll_l_c = assembly.component(sv_aroll_l,
                                  placement={"position": sv_aroll_o_l},
                                  label="sv_aroll_l")
sv_aroll_r_c = assembly.component(sv_aroll_r,
                                  placement={"position": sv_aroll_o_r},
                                  label="sv_aroll_r")


def revolute(first, first_o, second, second_o, place, point, axis_q,
             limits, label):
    """One posed world frame, expressed in each component's own frame.

    Both connectors are the SAME frame in world, which is what makes the
    residual zero and stops the solver rewriting the pose."""
    world = at(place, point)
    rotation = _q_mul(place[0], axis_q)
    return assembly.joint(
        "revolute",
        assembly.connector(first, "origin",
                           offset={"position": minus(world, first_o),
                                   "rotation": rotation}),
        assembly.connector(second, "origin",
                           offset={"position": minus(world, second_o),
                                   "rotation": rotation}),
        angle_limits_degrees=limits, label=label)


# Both endpoints of every limit are declared, always: api.task derives an
# action range from them, and a one-sided limit is filled in with a
# hundred-turn solver margin nobody designed. These are the ranges the
# parameter sliders already advertise, which is the mechanism's own answer.
HIP_ROLL_LIMITS = [-30.0, 45.0]
HIP_PITCH_LIMITS = [-60.0, 90.0]
KNEE_LIMITS = [-5.0, 130.0]
ANKLE_LIMITS = [-40.0, 40.0]
# +-20 deg of ankle roll, and it is a MECHANISM number rather than a
# preference: the sole is 40 mm wide, so its edge is 20 mm from the ankle's
# centreline and the roll axis is 6 mm above the sole. Rolling much past
# 20 deg lifts the inner edge of a foot that is standing on it, which is a
# way to fall over rather than a way to recover. Both endpoints declared,
# for the reason above.
ANKLE_ROLL_LIMITS = [-20.0, 20.0]

hip_roll_l = revolute(pelvis_c, pelvis_o, hip_l_c, hip_o_l,
                      NEUTRAL, (-hx, 0.0, roll_z), ROLL_Z,
                      HIP_ROLL_LIMITS, "hip_roll_l")
hip_roll_r = revolute(pelvis_c, pelvis_o, hip_r_c, hip_o_r,
                      NEUTRAL, (hx, 0.0, roll_z), ROLL_Z,
                      HIP_ROLL_LIMITS, "hip_roll_r")
hip_pitch_l = revolute(hip_l_c, hip_o_l, thigh_l_c, thigh_o_l,
                       place_l[0], (-hx, 0.0, hip_z), PITCH_Z,
                       HIP_PITCH_LIMITS, "hip_pitch_l")
hip_pitch_r = revolute(hip_r_c, hip_o_r, thigh_r_c, thigh_o_r,
                       place_r[0], (hx, 0.0, hip_z), PITCH_Z,
                       HIP_PITCH_LIMITS, "hip_pitch_r")
knee_l = revolute(thigh_l_c, thigh_o_l, calf_l_c, calf_o_l,
                  place_l[1], (-hx, 0.0, knee_z), PITCH_Z,
                  KNEE_LIMITS, "knee_l")
knee_r = revolute(thigh_r_c, thigh_o_r, calf_r_c, calf_o_r,
                  place_r[1], (hx, 0.0, knee_z), PITCH_Z,
                  KNEE_LIMITS, "knee_r")
ankle_l = revolute(calf_l_c, calf_o_l, brk_l_c, brk_o_l,
                   place_l[2], (-hx, 0.0, ankle_z), PITCH_Z,
                   ANKLE_LIMITS, "ankle_l")
ankle_r = revolute(calf_r_c, calf_o_r, brk_r_c, brk_o_r,
                   place_r[2], (hx, 0.0, ankle_z), PITCH_Z,
                   ANKLE_LIMITS, "ankle_r")
# The joint this whole slice is for. Lateral recovery on the old machine was
# hip_roll and weight shift and nothing else, and `compare.py` measured what
# that is worth: the `lat` column was the worst of the three at every shove
# scale, which ADR-087 predicted and called a MECHANISM finding. This is the
# mechanism answering it.
ankle_roll_l = revolute(brk_l_c, brk_o_l, foot_l_c, foot_o_l,
                        place_l[3], (-hx, 0.0, aroll_z), ROLL_Z,
                        ANKLE_ROLL_LIMITS, "ankle_roll_l")
ankle_roll_r = revolute(brk_r_c, brk_o_r, foot_r_c, foot_o_r,
                        place_r[3], (hx, 0.0, aroll_z), ROLL_Z,
                        ANKLE_ROLL_LIMITS, "ankle_roll_r")


# The ball-of-foot hinge is WELDED for standing. It removes two floppy
# unactuated degrees of freedom a stationary task has no use for; the pin, the
# notch and the tongue are all still drawn, so unwelding it for walking is one
# word. Foot and toe share a pose, so this frame needs no axis.
def weld(first, first_o, second, second_o, place, point, label):
    world = at(place, point)
    return assembly.joint(
        "fixed",
        assembly.connector(first, "origin",
                           offset={"position": minus(world, first_o)}),
        assembly.connector(second, "origin",
                           offset={"position": minus(world, second_o)}),
        label=label)


toe_l_j = weld(foot_l_c, foot_o_l, toe_l_c, toe_o_l,
               place_l[4], (-hx, ball_y, ball_r), "toe_l")
toe_r_j = weld(foot_r_c, foot_o_r, toe_r_c, toe_o_r,
               place_r[4], (hx, ball_y, ball_r), "toe_r")

# A servo is bolted to its carrier, not hinged to it: eight more welds, each
# at that servo's own output axis. They add no degree of freedom -- they exist
# so an MG90S can be made of something other than the bracket holding it.
sv_roll_l_j = weld(pelvis_c, pelvis_o, sv_roll_l_c, sv_roll_o_l,
                   NEUTRAL, (-hx, 0.0, roll_z), "sv_roll_l")
sv_roll_r_j = weld(pelvis_c, pelvis_o, sv_roll_r_c, sv_roll_o_r,
                   NEUTRAL, (hx, 0.0, roll_z), "sv_roll_r")
sv_pitch_l_j = weld(hip_l_c, hip_o_l, sv_pitch_l_c, sv_pitch_o_l,
                    place_l[0], (-hx, 0.0, hip_z), "sv_pitch_l")
sv_pitch_r_j = weld(hip_r_c, hip_o_r, sv_pitch_r_c, sv_pitch_o_r,
                    place_r[0], (hx, 0.0, hip_z), "sv_pitch_r")
sv_knee_l_j = weld(thigh_l_c, thigh_o_l, sv_knee_l_c, sv_knee_o_l,
                   place_l[1], (-hx, 0.0, knee_z), "sv_knee_l")
sv_knee_r_j = weld(thigh_r_c, thigh_o_r, sv_knee_r_c, sv_knee_o_r,
                   place_r[1], (hx, 0.0, knee_z), "sv_knee_r")
sv_ankle_l_j = weld(calf_l_c, calf_o_l, sv_ankle_l_c, sv_ankle_o_l,
                    place_l[2], (-hx, 0.0, ankle_z), "sv_ankle_l")
sv_ankle_r_j = weld(calf_r_c, calf_o_r, sv_ankle_r_c, sv_ankle_o_r,
                    place_r[2], (hx, 0.0, ankle_z), "sv_ankle_r")
sv_aroll_l_j = weld(brk_l_c, brk_o_l, sv_aroll_l_c, sv_aroll_o_l,
                    place_l[3], (-hx, 0.0, aroll_z), "sv_aroll_l")
sv_aroll_r_j = weld(brk_r_c, brk_o_r, sv_aroll_r_c, sv_aroll_o_r,
                    place_r[3], (hx, 0.0, aroll_z), "sv_aroll_r")

SERVO_COMPONENTS = [sv_roll_l_c, sv_roll_r_c, sv_pitch_l_c, sv_pitch_r_c,
                    sv_knee_l_c, sv_knee_r_c, sv_ankle_l_c, sv_ankle_r_c,
                    sv_aroll_l_c, sv_aroll_r_c]

asm = assembly.assembly(
    [pelvis_c, hip_l_c, hip_r_c, thigh_l_c, thigh_r_c, calf_l_c, calf_r_c,
     brk_l_c, brk_r_c, foot_l_c, foot_r_c, toe_l_c,
     toe_r_c] + SERVO_COMPONENTS + [ground],
    [hip_roll_l, hip_roll_r, hip_pitch_l, hip_pitch_r, knee_l, knee_r,
     ankle_l, ankle_r, ankle_roll_l, ankle_roll_r, toe_l_j, toe_r_j,
     sv_roll_l_j, sv_roll_r_j, sv_pitch_l_j, sv_pitch_r_j,
     sv_knee_l_j, sv_knee_r_j, sv_ankle_l_j, sv_ankle_r_j,
     sv_aroll_l_j, sv_aroll_r_j],
    label="mg_legs")
diag = assembly.solve(asm)


# ---------------------------------------------------------------------------
# Mass, and what may touch what.
# ---------------------------------------------------------------------------
#
# The floor's collision box is OFFSET by half its height. A collision
# primitive sits in the component frame and `part.box`'s origin is a corner,
# so the same extents with no offset would put the colliding top 20 mm above
# the floor anyone can see (ADR-074).
#
# ONLY THE SOLES COLLIDE. Thighs, calves, pelvis and hip links get no
# collision shape, so they touch nothing -- which is right for standing, keeps
# ngeom at five, and keeps the contact solver out of the mechanism.
# TWO DENSITIES, because this machine is two materials.
#
# PLA for everything printed: 1240 kg/m3 is solid PLA, and these brackets are
# 2.5 mm plate that prints essentially solid, so no infill discount is taken.
# It is the heavier of the two defensible assumptions and therefore the safe
# one for an actuator study.
#
# The servos are not printed. An MG90S is 13.4 g of coreless motor, brass
# gearset and case in about 7.3 cm3, and SERVO_DENSITY is set so the exported
# body weighs that -- checked by `measure.py`, not asserted here. Aluminium
# survives only as the floor, where mass is meaningless because it is
# grounded.
PLA = 1240.0
SERVO_DENSITY = 1891.0
FLOOR = 2700.0
heel_y = 0.35 * p.foot_len
toe_y0 = ball_y + ball_r + toe_clr
toe_y1 = 0.65 * p.foot_len


def pad(place, origin, x, y0, y1, width):
    """One box under a sole, from the drawing's own footprint bounds."""
    centre = at(place, (x, 0.5 * (y0 + y1), 0.5 * foot_h))
    return assembly.collision(
        "box", size_mm=[width, y1 - y0, foot_h],
        offset={"position": minus(centre, origin), "rotation": place[0]},
        friction=1.2)


bodies = [
    # A PLANE, since B2, where this was a box with a 20 mm offset.
    #
    # The offset was there because a box's colliding surface is its top face
    # and `part.box`'s origin is a corner (ADR-074). A plane's surface IS its
    # own origin, so the offset goes away with it -- and the ground component
    # is at the world origin, so the floor is exactly z=0 with nothing to get
    # wrong. Its `size_mm` is two full widths and a display grid spacing
    # rather than three extents; 0 would be a floor with no edge at all, and
    # 1200 mm is kept so the drawing still shows where the floor is.
    #
    # The reason it is worth changing at all is ADR-103's measurement: with a
    # box floor, MJX and stock MuJoCo disagreed about the contact count on a
    # fifth of all steps and about the state by 2.3e-7 per step; with a plane
    # the same comparison is 4.4e-16. The trainer and this engine are then
    # simulating the same machine, which is the one thing a policy trained
    # over there and played over here depends on.
    #
    # The ground's SOLID is still the box you can see: mass comes from a
    # body's solids and never from its collision shapes, so a plane costs it
    # nothing -- and the floor is grounded, where mass is meaningless anyway.
    assembly.body(ground, density_kg_m3=FLOOR,
                  collision=assembly.collision(
                      "plane", size_mm=[1200.0, 1200.0, 50.0],
                      friction=1.0)),
    assembly.body(pelvis_c, density_kg_m3=PLA),
    assembly.body(hip_l_c, density_kg_m3=PLA),
    assembly.body(hip_r_c, density_kg_m3=PLA),
    assembly.body(thigh_l_c, density_kg_m3=PLA),
    assembly.body(thigh_r_c, density_kg_m3=PLA),
    assembly.body(calf_l_c, density_kg_m3=PLA),
    assembly.body(calf_r_c, density_kg_m3=PLA),
    assembly.body(brk_l_c, density_kg_m3=PLA),
    assembly.body(brk_r_c, density_kg_m3=PLA),
    assembly.body(foot_l_c, density_kg_m3=PLA,
                  collision=pad(place_l[4], foot_o_l, -hx, -heel_y, ball_y,
                                p.foot_w)),
    assembly.body(foot_r_c, density_kg_m3=PLA,
                  collision=pad(place_r[4], foot_o_r, hx, -heel_y, ball_y,
                                p.foot_w)),
    assembly.body(toe_l_c, density_kg_m3=PLA,
                  collision=pad(place_l[5], toe_o_l, -hx, toe_y0, toe_y1,
                                0.75 * p.foot_w)),
    assembly.body(toe_r_c, density_kg_m3=PLA,
                  collision=pad(place_r[5], toe_o_r, hx, toe_y0, toe_y1,
                                0.75 * p.foot_w)),
]
bodies = bodies + [assembly.body(item, density_kg_m3=SERVO_DENSITY)
                   for item in SERVO_COMPONENTS]

# ---------------------------------------------------------------------------
# Actuators. PROVISIONAL -- `measure.py` then `feasibility.py` set these.
# ---------------------------------------------------------------------------
#
# Torque motors rather than position servos, and that is the decision the task
# rests on: a position servo holds its setpoint by itself, so a policy that
# emits the reset pose stands without learning anything and the reward
# measures nothing. With torque control, zero action is zero torque and the
# machine collapses. It is the harder thing to learn and it is chosen
# deliberately.
#
# SIZED AS THE HARDWARE, not as the mechanism, and that is the decision.
#
# An MG90S stalls at about 1.8 kgf.cm at 4.8 V and 2.2 at 6 V -- 176 and
# 216 N*mm. The `legs` sizing this file started from (250/750/750/300) is up
# to 3.5x that, and a policy trained against it learns to command torque the
# robot on the bench cannot produce. So all eight are one number: what one
# MG90S delivers at 6 V.
#
# `feasibility.py` says this is not tight, it is generous. Holding the stance
# takes 2.39 N*mm -- one per cent of one servo -- and the hand-written PD
# stood a whole episode on a peak of 4.5.
#
# And the FOOT is what actually limits this robot, not the servo. The centre
# of pressure cannot leave the sole, which reaches 45.5 mm ahead of the
# ankle, so past 2.581 N x 45.5 mm = 117 N*mm the foot rolls instead of
# pushing. An MG90S delivers 216. There is 1.8x more servo here than the
# footprint can spend, and going to PLA is what opened that gap -- in
# aluminium, at 4.825 N, the two were matched to within 2%.
# ---------------------------------------------------------------------------
# RE-RATED TO CONTINUOUS DUTY (M9, ADR-086).
#
# 216 N*mm is MG90S **stall**: the torque it produces at zero speed, at the
# instant before it stops turning, drawing its maximum current. It is a
# momentary figure and no hobby servo holds it -- the winding heats, the
# thermal cutout trips or the gears go, and the datasheet does not say for
# how long because the answer is "not long".
#
# The first policy trained against it did exactly what that permits. ADR-083
# measured it holding `hip_pitch_l/r` and both knees between 93 % and 99 %
# of 216 N*mm for the **entire six-second episode** -- it did not balance,
# it braced, widening its stance from +-30.00 mm to +-37.2/37.4 mm and then
# holding that splay with torque. On the bench that is a machine that stands
# for a few seconds and then smells of hot plastic.
#
# 86 N*mm is ~40 % of stall, and it is an **engineering judgment rather than
# a datasheet number**: hobby servos publish no continuous rating, and 30-50
# % of stall is the range within which small RC servos are conventionally
# run for sustained holding. It is stated here rather than buried so that if
# the machine cannot stand at it, the number moves *with the reasoning
# recorded* and not quietly.
#
# It is still ~19x what standing actually costs. `mj_inverse` measures the
# static stance at 2.39 N*mm and the hand-written PD peaked at 4.5, so the
# 213 N*mm the trained policy was spending was entirely self-inflicted.
#
# A consequence worth stating, because it changes which constraint is
# active. ADR-082 found the FOOT limiting this robot at 117 N*mm: the centre
# of pressure cannot leave the sole, which reaches 45.5 mm ahead of the
# ankle, so past 2.581 N x 45.5 mm the foot rolls instead of pushing.
#
# At 216 a single ankle could out-torque the whole footprint by 1.8x, which
# means the machine could tip ITSELF by over-torquing one ankle. At 86 it
# cannot: one ankle is now below 117, so no single ankle command can roll
# the foot. `feasibility.py` check 3 measures where that leaves the machine
# overall -- the two ankles together still reach 172 N*mm, so for rejecting
# a shove the FOOT is still what binds at 117, with 1.87x margin over the
# 62.8 N*mm the declared shove needs.
MG90S_STALL_NMM = 216.0          # momentary, and what ADR-083 measured
MG90S_CONTINUOUS_NMM = 86.0      # ~40 % of stall; the judgment above
ROLL_TORQUE = MG90S_CONTINUOUS_NMM
PITCH_TORQUE = MG90S_CONTINUOUS_NMM
KNEE_TORQUE = MG90S_CONTINUOUS_NMM
ANKLE_TORQUE = MG90S_CONTINUOUS_NMM


def motor(joint, limit, label):
    return assembly.actuator(joint, kind="motor", control_nmm="0",
                             torque_limit_nmm=limit, label=label)


hip_roll_l_m = motor(hip_roll_l, ROLL_TORQUE, "hip_roll_l_m")
hip_roll_r_m = motor(hip_roll_r, ROLL_TORQUE, "hip_roll_r_m")
hip_pitch_l_m = motor(hip_pitch_l, PITCH_TORQUE, "hip_pitch_l_m")
hip_pitch_r_m = motor(hip_pitch_r, PITCH_TORQUE, "hip_pitch_r_m")
knee_l_m = motor(knee_l, KNEE_TORQUE, "knee_l_m")
knee_r_m = motor(knee_r, KNEE_TORQUE, "knee_r_m")
ankle_l_m = motor(ankle_l, ANKLE_TORQUE, "ankle_l_m")
ankle_r_m = motor(ankle_r, ANKLE_TORQUE, "ankle_r_m")
# The same MG90S and therefore the same 86 N*mm judgment. Nothing about
# adding a tenth motor changes what one servo can hold continuously, and
# giving the new joint a different number would be inventing a servo.
ankle_roll_l_m = motor(ankle_roll_l, ANKLE_TORQUE, "ankle_roll_l_m")
ankle_roll_r_m = motor(ankle_roll_r, ANKLE_TORQUE, "ankle_roll_r_m")

MOTORS = [hip_roll_l_m, hip_roll_r_m, hip_pitch_l_m, hip_pitch_r_m,
          knee_l_m, knee_r_m, ankle_l_m, ankle_r_m,
          ankle_roll_l_m, ankle_roll_r_m]

# MuJoCo's joints are frictionless, undamped and massless in their own rotors,
# which no real joint is -- least of all a geared hobby servo, whose rotor
# inertia is multiplied by the square of a ~270:1 gear ratio. Armature is what
# that reflected inertia is called, and on limbs this light it is the term
# that conditions a stiff torque loop.
DAMPS = [
    assembly.joint_dynamics(hip_roll_l, damping_nmms_per_deg=0.1,
                            armature_kgmm2=20.0),
    assembly.joint_dynamics(hip_roll_r, damping_nmms_per_deg=0.1,
                            armature_kgmm2=20.0),
    assembly.joint_dynamics(hip_pitch_l, damping_nmms_per_deg=0.1,
                            armature_kgmm2=20.0),
    assembly.joint_dynamics(hip_pitch_r, damping_nmms_per_deg=0.1,
                            armature_kgmm2=20.0),
    assembly.joint_dynamics(knee_l, damping_nmms_per_deg=0.1,
                            armature_kgmm2=10.0),
    assembly.joint_dynamics(knee_r, damping_nmms_per_deg=0.1,
                            armature_kgmm2=10.0),
    assembly.joint_dynamics(ankle_l, damping_nmms_per_deg=0.05,
                            armature_kgmm2=5.0),
    assembly.joint_dynamics(ankle_r, damping_nmms_per_deg=0.05,
                            armature_kgmm2=5.0),
    # The ankle roll carries the same servo through the same gearbox onto a
    # link of the same order of inertia as the ankle pitch's, so it takes the
    # ankle pitch's row rather than a number of its own.
    assembly.joint_dynamics(ankle_roll_l, damping_nmms_per_deg=0.05,
                            armature_kgmm2=5.0),
    assembly.joint_dynamics(ankle_roll_r, damping_nmms_per_deg=0.05,
                            armature_kgmm2=5.0),
]


# ---------------------------------------------------------------------------
# What the policy sees: 52 channels of the 64 a task may declare.
# ---------------------------------------------------------------------------
#
# The pelvis is the free root, so `centre_of_mass` read off it is the WHOLE
# machine's centre of mass -- the one quantity a balance reward actually
# wants.
#
# There is no projected-gravity channel and none is needed: the world-Z
# component of the pelvis's own +Z is 1 - 2*(pel_qx^2 + pel_qy^2), which is
# arithmetic a reward formula may write. It is 1 upright, 0 on its side, and
# it is the backbone of both the tilt cost and the fall termination.
#
# The two FEET are new in M9b (ADR-087), and they are six channels bought for
# one reason: the task stopped being "stand still" and became "take a step if
# you have to", and where a foot is relative to the centre of mass IS the
# state variable a stepping recovery is written in. A policy can in principle
# infer foot placement from eight joint angles and a base pose -- it is
# forward kinematics, and the network is big enough -- but making it derive
# the one quantity the new reward is about, every step, from scratch, is
# spending capacity on arithmetic. 46 of 64 left room and this is what the
# room was for.
#
# 52 since B2: the two ankle-roll joints bring an angle, a velocity and a
# torque each, and every one of the three is read the same way the other
# eight joints' are. The room is still there -- 52 of 64.
#
# 55 since B6 (ADR-112), and the three are the reason the engine grew a
# ninth observation kind. `centre_of_mass_velocity` is `mjSENS_SUBTREELINVEL`
# off the free root: the velocity of the WHOLE machine's centre of mass, not
# of one link's frame origin. The distinction is not pedantry and it is not
# free to get wrong -- `pv`, the pelvis's own frame velocity, is already
# declared three lines up, and measured over 500 randomised states at
# realistic recovery speeds it differs from the true centre-of-mass velocity
# by 19%. That is 9 mm of capture-point error at 400 mm/s and 18 mm at
# 850 mm/s, against a 24.5 mm backward support margin: 40-75% of the very
# quantity the new reward term is built to compare against. Reaching for `pv`
# because it was already there is the ADR-107 mistake in a new place.
#
# `pv` STAYS. It is a different thing and the tilt/stillness terms read it;
# what changes is that the term deciding "must I step?" no longer does.
#
# 58 since B7 (ADR-116), and the three are the reason the engine grew a
# TENTH observation kind. `centroidal_angular_momentum` is
# `mjSENS_SUBTREEANGMOM` off the same free root: the angular momentum of the
# WHOLE machine about its own centre of mass, in N*mm*s.
#
# It is here because of what B6 died of. Every death at every force level on
# `capability.py`'s sweep was `tipped`; not one was `collapsed`. Tipping is
# rotational, and NOTHING in the reward could see rotational momentum:
# `tilt` prices `pel_qx^2 + pel_qy^2`, which is the pelvis's ORIENTATION,
# and `spin` prices `pw_*`, which is that one link's FRAME angular rate. A
# machine can be perfectly upright, its pelvis perfectly still, and already
# going over -- the legs carry the momentum that takes it over and both
# terms read approximately nothing while it happens. That is the state a
# stepping recovery has to be triggered from and six runs could not see it.
#
# The distinction is the same one `cv` had to be bought for, one derivative
# up in the rotational half, and it is measured rather than argued: on the
# arm fixture, kicking the child link alone, the parent's FRAME angular
# velocity reads exactly 0.0 where the subtree carries 18.89 N*mm*s. The two
# are not proportional either -- the ratio moves by a factor of six over one
# swing -- so this is not `pw` with a different weight on it.
#
# `pw` STAYS, for the reason `pv` did: `spin` reads it and it is a different
# quantity. What is new is that the term pricing the machine's rotation no
# longer has to be written on it.
JOINTS = [hip_roll_l, hip_roll_r, hip_pitch_l, hip_pitch_r,
          knee_l, knee_r, ankle_l, ankle_r, ankle_roll_l, ankle_roll_r]
NAMES = ["hip_roll_l", "hip_roll_r", "hip_pitch_l", "hip_pitch_r",
         "knee_l", "knee_r", "ankle_l", "ankle_r",
         "ankle_roll_l", "ankle_roll_r"]

CHANNELS = [
    assembly.observation(pelvis_c, "component_position", name="pel"),
    assembly.observation(pelvis_c, "component_orientation", name="pel"),
    assembly.observation(pelvis_c, "component_linear_velocity", name="pv"),
    assembly.observation(pelvis_c, "component_angular_velocity", name="pw"),
    assembly.observation(pelvis_c, "centre_of_mass", name="com"),
    assembly.observation(pelvis_c, "centre_of_mass_velocity", name="cv"),
    assembly.observation(pelvis_c, "centroidal_angular_momentum", name="cam"),
    assembly.observation(foot_l_c, "component_position", name="ft_l"),
    assembly.observation(foot_r_c, "component_position", name="ft_r"),
]
for index in range(len(JOINTS)):
    CHANNELS.append(assembly.observation(JOINTS[index], "position",
                                         name=NAMES[index] + "_a"))
for index in range(len(JOINTS)):
    CHANNELS.append(assembly.observation(JOINTS[index], "velocity",
                                         name=NAMES[index] + "_v"))
for index in range(len(MOTORS)):
    CHANNELS.append(assembly.observation(MOTORS[index], "actuator_force",
                                         name=NAMES[index] + "_tau"))

model = assembly.mjcf(asm, bodies, actuators=MOTORS, joint_dynamics=DAMPS,
                      observations=CHANNELS, label="mg_legs_model")


# ---------------------------------------------------------------------------
# The task: stay standing, and step if that is what it takes (M9b, ADR-087).
# ---------------------------------------------------------------------------
#
# It was "stand still, and keep standing" through M9, and "still" is the word
# that changed. The M9 reward paid for stillness so heavily that a machine
# knocked off its feet scored better by going down than by moving 100 mm and
# coming back -- the arithmetic is below the weights. This one pays for the
# recovery.
#
# EVERY TERM READS ~0 AT THE STANDING POSE, which is hazard 9 and is not a
# style preference: the same reward written against an absolute channel with a
# large offset trained WORSE THAN NOT TRAINING (4.46 -> 3.66) where the
# displacement form went -0.243 -> -0.028. So the baselines below are
# subtracted in the expression, and they are MEASURED rather than read off the
# drawing.
#
# MEASURED, by `measure.py`, off the exported keyframe of THIS machine.
#
# These three are the ONLY place the standing pose is written down, and every
# expression below is built from them by string concatenation rather than by
# repeating the number. That is not tidiness: through M9c the same figure was
# typed into the height term, into DISPLACEMENT and into the `collapsed`
# floor, and three copies of a measured constant is three chances for the
# reward to be about a machine that no longer exists. B2 changes the drawing,
# so all three move, and they now move once.
#
# RE-MEASURE THEM AFTER ANY CHANGE TO THE DRAWING. Getting this wrong trains
# WORSE than not training -- hazard 9, measured: the same reward written
# against an absolute channel with a large offset went 4.46 -> 3.66 where the
# displacement form went -0.243 -> -0.028.
# B2, measured: 302.01 g -- 263.07 g of it was the M9c machine -- standing
# 144.210 mm to its centre of mass. The centre of mass came DOWN by 1.8 mm
# even though every joint above the ankle went UP by 20 mm, because the two
# new servos sit at z = 19 mm and the two brackets at z = 34, which is the
# lowest place on this machine anything can be bolted. omega0 = sqrt(g/h)
# is 8.25 rad/s against the old 8.20, so the capture-point arithmetic in the
# task below survives the rebuild almost unchanged.
Z0 = 144.210
X0 = 0.002
Y0 = 1.225

# B6 (ADR-112). Measured off this bundle's own MJCF at the reset keyframe,
# the same way Z0/X0/Y0 were:
#
#     foot frame-origin centroid       (0.000, 12.250) mm
#     standing CoM minus that centroid (0.002, -11.025) mm
#     omega0 = sqrt(9810 / 144.210)    8.2478 rad/s
#
# The centroid is not the same point as the CoM -- the machine stands with
# its centre of mass 11 mm BEHIND the midpoint of its two foot frames -- so
# the offset is carried explicitly. That is hazard 9 and it is the difference
# this file has already measured between training and training worse than not
# training: both expressions below read EXACTLY 0 at the standing pose.
FX0, FY0 = 0.000, 12.250
OMEGA0 = 8.2478
FOOT_X = "((ft_l_x + ft_r_x) / 2.0)"
FOOT_Y = "((ft_l_y + ft_r_y) / 2.0)"

EFFORT = " + ".join(["abs(" + name + "_tau)" for name in NAMES])
POSTURE = " + ".join(["abs(" + name + "_a)" for name in NAMES])
DISPLACEMENT = ("abs(com_x - (" + str(X0) + ")) + "
                "abs(com_y - (" + str(Y0) + "))")

# Where the centre of mass is RELATIVE TO THE FEET, zeroed at the standing
# pose. This is the reference change, and it is the whole of B6.
OVER = ("abs(com_x - " + FOOT_X + " - (0.002)) + "
        "abs(com_y - " + FOOT_Y + " - (-11.025))")

# The capture point, xi = p_com + v_com / omega0, against the same reference.
# Pratt's quantity, and the state variable the push-recovery literature
# writes a stepping controller in: it is where the centre of mass would have
# to be caught, so when it leaves the support polygon the only way to reduce
# this cost is TO MOVE A FOOT.
XI = ("abs(com_x + cv_x / " + str(OMEGA0) + " - " + FOOT_X + " - (0.002)) + "
      "abs(com_y + cv_y / " + str(OMEGA0) + " - " + FOOT_Y + " - (-11.025))")

# B7 (ADR-116). Two new expressions, both read off channels that are already
# zero when the machine is standing, so hazard 9 costs nothing to satisfy
# here: a machine at rest has no centre-of-mass velocity and no angular
# momentum, and `tanh(0) = 0`.
#
# ARREST is the TRUE speed of the centre of mass -- horizontal only, because
# the vertical component is a crouch absorbing a landing and taxing it is
# how M9 got a machine that would rather fall than move. This is what turns
# STEPPED into RECOVERED: after the catch the machine has to come to rest,
# and nothing in B6 paid for that. `stillness` is -0.0005 on the PELVIS
# FRAME velocity, which ADR-112 measured as 19% off the real thing and which
# is a hundredth the weight anyway.
#
# It is not redundant with `capture`. xi can sit exactly over the feet while
# the body is still travelling fast -- that is what "caught but not stopped"
# is -- because xi is a position and this is a rate.
ARREST = "abs(cv_x) + abs(cv_y)"

# SWIRL is the term the engine change exists for: the horizontal angular
# momentum whose accumulation ends every B6 episode as `tipped`.
#
# HORIZONTAL COMPONENTS ONLY, and that is a decision rather than symmetry
# with the line above. `cam_z` is yaw -- turning about the vertical -- and a
# sideways catch step is a yaw excursion by construction. Pricing it would
# tax the very thing this run is asking for.
SWIRL = "abs(cam_x) + abs(cam_y)"

# The scales. Every one of these is the value at which the term reaches
# tanh(1) = 0.762, so it is "the magnitude this term is ABOUT" and a number
# to put beside a measurement rather than to tune.
#
# CAPTURE_FINE / CAPTURE_FAR: see the reward block below. 25 mm is the
# polygon edge; 110 mm is most of a single step's reach.
#
# ARREST_SPEED at 300 mm/s: the machine's own recovery speeds. M9's stumble
# injects up to 250 mm/s at reset and a 1 N shove leaves it moving about
# 400 mm/s, so the term is sloped across exactly the band a recovery lives
# in rather than saturated above it.
#
# SWIRL_MOMENTUM is MEASURED, not chosen -- `swirl_scale.py`, off this
# bundle, with `feasibility.py` check 6's PD holding the stance while the
# DECLARED shove lands, eight azimuths by five force levels across the band:
#
#     |cam_xy| at the reset keyframe                        0.000000
#     0.40 N  median  0.98   p90   4.30   held 7/8
#     0.60 N  median  5.34   p90  31.96   went over 6/8
#     0.80 N  median 12.71   p90  36.10   went over 8/8
#     1.00 N  median 15.44   p90  38.99   went over 8/8
#     1.20 N  median 17.87   p90  39.62   went over 8/8
#
# The median over the levels that actually go over -- 0.60 N and up, which
# is the regime this term is about -- is **13.66 N*mm*s** over 24064
# samples. 13.7 is that, rounded.
#
# IT IS RE-MEASURED WHENEVER THE BAND MOVES, and this run is why that is a
# rule rather than a nicety: at the original [0.40, 2.50] band the same
# procedure said 16.91, and carrying that number down to a [0.40, 1.20]
# band would have left the term reading 20% flatter than intended across
# every recovery it is supposed to price.
#
# THE PLACEHOLDER WAS 8.0 AND IT WOULD HAVE BEEN WRONG IN THE EXPENSIVE
# DIRECTION. tanh(13.66 / 8.0) = 0.937: the term would have been saturated
# across most of the recovery regime, which is a constant rather than a
# gradient and is precisely the defect that split `capture` in two.
# Measuring a scale before fixing a weight costs a minute; not measuring one
# costs a run.
#
# Note also what the 0.40 N row says: at the absorbable end of the band the
# median is under 1 N*mm*s, so `swirl` is nearly silent on an episode the
# machine can answer with its ankles. That is the shape wanted -- it prices
# a machine that is going over, not a machine that is moving.
CAPTURE_FINE = 25.0
CAPTURE_FAR = 110.0
ARREST_SPEED = 300.0
SWIRL_MOMENTUM = 13.7

# M9b (ADR-087): the weights below are not the M9 ones tuned, they are the M9
# ones ARITHMETICALLY REFUTED. Price out one step of a stumble under the old
# numbers -- centre of mass 100 mm out, moving at 800 mm/s, tilted 20 deg,
# which is exactly the state a machine catching itself passes through:
#
#     over_feet   -0.02 /mm    x 100 mm      = -2.00
#     stillness   -0.002 /mm/s x 800 mm/s    = -1.60
#     height      -0.02 /mm    x  30 mm      = -0.60
#     posture     -0.004 /deg  x 100 deg     = -0.40
#     spin, tilt, effort                     = -0.60
#     alive                                  = +1.00
#                                              -----
#                                              -4.20 per step
#
# Falling over immediately forgoes the +1 alive bonus and NOTHING ELSE.
# Stumbling for 150 steps and then recovering scores worse than going down at
# once, so the previous policy was not refusing to step: it was correctly
# declining a bad deal. The same arithmetic under the weights below comes to
# about -0.9 per step, which makes catching yourself worth roughly +400 over
# the episode against 0 for falling. That reversal is the whole change.
#
# The one term that had to change SHAPE rather than weight is `over_feet`.
# Linear in displacement, it forbids stepping: every millimetre out costs the
# same, so a 200 mm excursion is twenty times a 10 mm one and there is no
# distance at which "get back" beats "go down". Two terms replace it:
#
#   over_feet  tanh(d / 40 mm), SATURATING -- steep inside the support
#              polygon, flat past about 80 mm. Being far out is bad; being
#              twice as far out is not twice as bad, because from either one
#              the answer is the same step.
#   drift      linear and tiny, so there is still a gradient pointing home
#              from anywhere. This is the guard against wandering off
#              (ADR-087 risk): if the machine walks away rather than
#              recovering, this is the weight to raise, NOT over_feet's
#              shape to revert.
#
# `splay` is new and is the ADR-083 brace priced where it actually lived. The
# braced policy did not stiffen everything; it drove the two hip rolls apart
# and stood on a widened stance. Loosening `posture` by 4x to let the legs
# move would have made that free again, so the roll pair keeps its old price
# while the six joints a recovery needs get cheap.
#
# EVERY TERM STILL READS 0 AT THE STANDING POSE (hazard 9): tanh(0) = 0, and
# the standing hip rolls are 0.
#
# ---------------------------------------------------------------------------
# B6 (ADR-112): NO REWARD TERM IN THIS FILE HAD EVER NAMED THE FEET.
#
# Five runs -- m9c, B2, B3, B4, B5 -- produced a machine that stands, absorbs
# a shove with its joints, and never lands a recovery step. B3, B4 and B5
# each changed the disturbance on the hypothesis that the task never ASKED
# for one. B5 settled that: aimed backward at [210, 330] deg in the band
# [0.30, 0.80] N, checkpoint 750 puts a foot out in 4 of 6 episodes, timed
# 0.09-0.12 s after the push. It steps. It does not land.
#
# Read the reward against the channels and the reason is plain. `ft_l_*` and
# `ft_r_*` were bought in M9b for one stated reason -- "where a foot is
# relative to the centre of mass IS the state variable a stepping recovery is
# written in", nine hundred lines up -- and NO TERM NAMED THEM. Both spatial
# terms measured the centre of mass against (X0, Y0): the fixed point on the
# floor where the machine happened to stand at t=0. So:
#
#   * moving a FOOT changed the reward by exactly nothing;
#   * standing successfully at a new place after a step was penalised FOR THE
#     REST OF THE EPISODE -- at 60 mm out, -0.5*tanh(60/40) - 0.002*60
#     = -0.45 - 0.12 = -0.57 per step against the +1.00 `alive` pays.
#
# A machine optimising that is not declining to step out of cowardice. It is
# correctly reading an objective that says KEEP YOUR CENTRE OF MASS OVER THIS
# ONE SPOT ON THE FLOOR, and the only actuators that can do that without
# moving the feet are the joints. Which is exactly what five runs produced.
#
# Two changes, and they are the same change twice:
#
#   over_feet  reference moves from the world origin to the foot centroid.
#              A step no longer costs anything once it lands.
#   capture    NEW, and the mechanism. tanh(xi / 40 mm) where xi is the
#              capture point against that same moving reference.
#
# WHY THIS AND NOT A BIGGER PUSH. Price the same backward shove both ways.
# Under B5, a capture point 40 mm behind the feet costs nothing directly --
# no term can see it -- and the payoff for stepping arrives 1-2 s later as
# survival, through a discount factor that at 100 Hz could barely reach it.
# Under B6 it costs -0.8 * tanh(1.0) = -0.61 PER STEP, RIGHT NOW, and putting
# a foot back under it zeroes that IMMEDIATELY. The delayed benefit becomes an
# immediate one. That is the whole mechanism, and it is what the 2026 DCM/RL
# work reports: ankle, hip and stepping strategies emerge in that order from
# one objective, because when xi leaves the polygon there is no other way to
# reduce the cost. Their ablation is the relevant number -- remove the
# balance-metric structure and the "stuck low" failure rate goes 0.067 -> 1.0.
#
# `drift` goes -0.002 -> -0.003. It is now the ONLY term that objects to
# wandering off, since `over_feet` follows the feet wherever they go, and it
# is what ADR-087 already named as the guard to raise if the machine walks
# away rather than recovering. Note what it costs a COMPLETED step: 60 mm out
# is -0.18 per step where B5 charged -0.57.
#
# `splay` goes -0.004 -> -0.001. ADR-083 priced it against a braced wide
# stance, which scored well because widening the feet did not change a
# world-referenced `over_feet`. It changes a foot-referenced one -- widening
# the stance moves the centroid the term is measured against -- so the
# exploit is now priced by the term's own shape, and the old -0.004 was
# taxing the lateral half of the very step being asked for.
#
# THE RISK, NAMED SO IT GETS CHECKED: SQUATTING. Both new terms are perfectly
# satisfied by a deep crouch, and B5 already showed deaths shifting from
# `tipped` to `collapsed`. The only guard is `height` at -0.010/mm -- -0.60
# per step at a 60 mm crouch, against the +1.00 `alive` pays -- and the
# 72.1 mm `collapsed` floor. Check mean `com_z` and the tipped/collapsed
# split AT THE FIRST CHECKPOINT, not at the end. If it is squatting, `height`
# is the weight to raise.
#
# ---------------------------------------------------------------------------
# B7 (ADR-116): 12 terms of the 16 a task may declare.
# ---------------------------------------------------------------------------
#
# THE REWARD BUDGET IS THE FIRST THING TO CHECK AND IT IS NOT OPTIONAL.
# `alive` pays +1.0 a step and every other term is a cost, so if the costs
# sum past +1.0 in the states the machine actually reaches, the per-step
# reward is negative, ENDING THE EPISODE BEATS CONTINUING IT, and the
# optimal policy is to fall over immediately. PPO finds that quickly and it
# reads on the training curve as a failure to learn.
#
# Measured on the states B6's own policy visits -- 12 seeds, 1263 states,
# `reward_decompose.py`:
#
#     B6's objective          +0.0103 per step   <-- 1% of headroom
#     B7 as first written     -0.2060            <-- pays to die
#     B7 as it stands now     +0.0327
#
# B6 SUCCEEDED ON ONE PERCENT. That is the number to keep in mind before
# adding anything here: there was never room for three new costs, and the
# first draft of B7 spent 22x the whole margin. Run `reward_decompose.py`
# against the last good policy before dispatching anything that adds a term.
#
# B6 worked -- 6 of 12 episodes step AND survive, on a number that had been
# zero in every run this project has ever done. What it did not do is
# RECOVER from anything big: its top shove is 0.80 N, xi 39 mm against a
# 24.5 mm heel margin, which is a step the eye can barely see. B7 asks for a
# big force, a visible catch step, and a return to rest. Three changes, each
# with a stated mechanism, and TEN TERMS UNCHANGED so the comparison against
# B6 stays readable.
#
#   (i)  `capture` -0.8 SPLIT into `capture_fine` -0.6 and `capture_far`
#        -0.6. B6's single tanh(xi/40) is SATURATED across the entire new
#        band -- tanh(120/40) = 0.995 against tanh(80/40) = 0.964 -- so
#        exactly where a step is the only answer there is no gradient. Two
#        scales, 25 mm and 110 mm, restore a slope at both ends. Combined
#        weight goes up, -0.8 -> -1.2, because this is the term the run is
#        about.
#
#   (ii) `arrest` -0.3, NEW. tanh((|cv_x| + |cv_y|) / 300 mm/s) on the TRUE
#        centre-of-mass speed. This is what turns STEPPED into RECOVERED:
#        after the catch the machine has to come to rest and nothing in B6
#        paid for that. Not redundant with `capture` -- xi is a position and
#        this is a rate, so xi can sit over the feet while the body is still
#        travelling.
#
#   (iii) `swirl` -0.3, NEW, and the term the engine change exists for.
#        tanh((|cam_x| + |cam_y|) / K) on the machine's own angular momentum
#        about its own centre of mass. Every B6 death is `tipped` and not one
#        is `collapsed`; this is the first term that can see the quantity
#        that ends them. K is MEASURED off this bundle, not chosen.
#
# TERMINATION IS UNCHANGED, deliberately and for two reasons. `tipped` above
# 0.15 on qx^2+qy^2 is a 45 deg lean, and at 45 deg this machine's capture
# point is already past the 180 mm a single step can reach -- so it is close
# to a physical bound rather than a safety margin. And keeping it keeps the
# survival number directly comparable to B6's 6/12, which is the whole basis
# for reading this run.
#
# HELD IN RESERVE, NOT IN THIS RUN: a both-feet-airborne hop guard -- which
# has to be written (ft_l_z - 8.6)^2 * (ft_r_z - 8.6)^2, since the expression
# vocabulary has no min() -- and relaxing `tipped`. B6 showed no hopping, and
# every added term is a new exploit surface.
#
# THE RISK, AGAIN AND UNCHANGED: both new spatial terms are still perfectly
# satisfied by a deep crouch, and `swirl` adds a second way to be trivially
# satisfied -- angular momentum is minimised by NOT MOVING. If the run comes
# back standing rigidly still and refusing to step, `swirl` is over-weighted
# against `capture` and those two are meant to trade.
REWARD = [
    assembly.reward("1", weight=1.0, label="alive"),
    # 1 - 2*(qx^2 + qy^2) is the upright signal; written as the deviation from
    # it so the term is 0 standing rather than 1. Unchanged: this is the
    # uprightness backbone and nothing about stepping makes falling over
    # cheaper.
    assembly.reward("pel_qx^2 + pel_qy^2", weight=-4.0, label="tilt"),
    # Halved. A stumble dips the centre of mass and absorbing a 45 mm drop
    # dips it further; at -0.02 that dip cost more than the fall.
    assembly.reward("abs(com_z - (" + str(Z0) + "))",
                    weight=-0.010, label="height"),
    # B6: measured against the FEET, not against the floor.
    assembly.reward("tanh((" + OVER + ") / 40.0)",
                    weight=-0.4, label="over_feet"),
    # B7: the capture point, ACROSS TWO SCALES. B6 wrote this as one term,
    # `tanh(xi / 40)` at -0.8, and the new shove band is what breaks it:
    # tanh(120/40) = 0.995 against tanh(80/40) = 0.964, so the term is
    # SATURATED across the whole big-push regime and there is no gradient
    # left exactly where a step is the only answer. A saturated term is not
    # a strong term, it is a constant.
    #
    # So the same quantity is priced twice at two lengths, and the pair asks
    # two different questions:
    #
    #   capture_fine  tanh(xi/25)   sharp at the polygon edge: MUST I STEP?
    #   capture_far   tanh(xi/110)  still sloped at 120 mm: HOW BIG A STEP?
    #
    # Combined -1.2 against B6's -0.8, because this is the term the run is
    # about. Both read exactly 0 at the standing pose for the same reason
    # the single term did -- same expression, same measured offsets.
    assembly.reward("tanh((" + XI + ") / " + str(CAPTURE_FINE) + ")",
                    weight=-0.6, label="capture_fine"),
    assembly.reward("tanh((" + XI + ") / " + str(CAPTURE_FAR) + ")",
                    weight=-0.6, label="capture_far"),
    # B7: come to REST after the catch. See ARREST above for why this is not
    # `stillness` and not `capture`.
    assembly.reward("tanh((" + ARREST + ") / " + str(ARREST_SPEED) + ")",
                    weight=-0.3, label="arrest"),
    # B7 (ADR-116): the machine's own angular momentum, which is what every
    # B6 episode ended on and what no term could see.
    assembly.reward("tanh((" + SWIRL + ") / " + str(SWIRL_MOMENTUM) + ")",
                    weight=-0.3, label="swirl"),
    # Still against the world origin, and now the only term that is: this is
    # the anti-wander guard and it is the one thing a foot-referenced
    # objective cannot express.
    assembly.reward(DISPLACEMENT, weight=-0.003, label="drift"),
    # `stillness` and `spin` ARE GONE, and B7a is the run that measured why.
    # They were:
    #
    #     stillness  -0.0005 * (abs(pv_x) + abs(pv_y) + abs(pv_z))
    #     spin       -0.0005 * (abs(pw_x) + abs(pw_y) + abs(pw_z))
    #
    # Each is replaced by the term that prices the same thing off the right
    # channel: `arrest` reads `cv`, the TRUE centre-of-mass velocity, where
    # `stillness` read `pv`, the pelvis frame velocity ADR-112 measured as
    # 19% off it; `swirl` reads `cam`, the whole machine's angular momentum,
    # where `spin` read `pw`, one link's frame angular rate. Keeping both
    # halves of each pair is paying twice for one quantity and paying the
    # worse of the two readings -- which is the argument ADR-112 and ADR-116
    # each already made, carried one step further than either did.
    #
    # And they are the only two UNBOUNDED terms in a reward that is
    # otherwise bounded tanh everywhere it shapes. That is not a symmetry
    # complaint. Measured at the states B7's own iteration-100 policy
    # visited, they were the two largest costs after `alive` pays --
    # -0.349 and -0.272 -- because an unbounded linear term grows without
    # limit exactly when the machine is in trouble, which is when the
    # shaping terms are supposed to be in charge.
    #
    # WHAT THIS COST, STATED IN FULL. B7 as first written was net-negative:
    # scored on the states B6's own policy visits -- the best machine this
    # project has -- B6's reward pays +0.0103 per step and B7's paid
    # -0.2060. A negative per-step reward with a termination available means
    # ENDING THE EPISODE BEATS CONTINUING IT, so the optimal policy is to
    # fall over at once. The 150-iteration probe found it: episode length
    # fell monotonically 23.5 -> 8.0 steps while loss converged to 2.46 and
    # sigma tightened. It did not fail to learn. It learned what it was
    # asked. Without these two terms the same states score +0.0327, which is
    # 3.2x the headroom B6 ran on.
    #
    # `pv` and `pw` STAY AS OBSERVATIONS. The policy reading a quantity and
    # the reward pricing it are different questions, and a frame velocity is
    # a perfectly good thing for a network to see.
    #
    # NOTE FOR THE NEXT RUN, not taken here: `drift` is now the largest
    # remaining unbounded term (-0.175 on the same states) and the second
    # largest cost overall. It stays because B6 had it and succeeded with
    # it, and changing two things at once is how a result stops being
    # attributable. It is the first candidate if headroom is short again.
    assembly.reward(EFFORT, weight=-0.0002, label="effort"),
    # Quartered: this pinned all eight joints to the solved pose, which is
    # the pose a step has to leave.
    assembly.reward(POSTURE, weight=-0.001, label="posture"),
    assembly.reward("abs(hip_roll_l_a) + abs(hip_roll_r_a)",
                    weight=-0.001, label="splay"),
]

# Tipped past about 45 degrees, or collapsed to half of standing height. Both
# end the episode as a FAILURE rather than a horizon, which is the difference
# the bundle records as `truncated` and a trainer learns from.
#
# `collapsed` goes 0.70 -> 0.60 (M9b) -> 0.50 (B3), and this time the
# evidence is direct rather than arithmetic. `capability.py` on the m9c
# policy: 8 of 12 deaths were `collapsed`, at a centre of mass of 77-87 mm
# against an 87.6 mm floor, with the tilt as low as 0.023 out of the 0.15
# that counts as tipped. The machine was UPRIGHT AND SINKING when the episode
# was ended -- which is exactly the state a recovery passes through on its
# way back up, so the floor was killing the thing it was meant to measure.
#
# The risk, stated because it is real: a lower floor makes SITTING DOWN a
# survivable strategy. The guard is the `height` term at -0.010/mm, and the
# arithmetic is worth doing rather than trusting. At 0.5*Z0 the machine may
# sink 83 mm before dying, and holding a 60 mm crouch costs -0.60 per step
# against the +1.00 `alive` pays -- so crouching is still worth doing to
# survive and still not worth doing for its own sake, which is the shape
# wanted. If a run comes back squatting, this weight is the one to raise.
#
# `tipped` does NOT move -- past about 45 degrees this machine is going down
# whatever it does, and pretending otherwise would just fill the batch with
# episodes spent falling.
TERMINATION = [
    assembly.termination("pel_qx^2 + pel_qy^2", above=0.15, label="tipped"),
    assembly.termination("com_z", below=0.5 * Z0, label="collapsed"),
]

# Mass to ten per cent, damping and armature to a quarter. With a fixed reset
# pose and no push, this is the whole of what stops one open-loop torque
# sequence from working across the batch.
#
# THE BUDGET IS 32 ENTRIES AND B2 ASKED FOR 41: two more links, two more
# servos, two more dampings and two more armatures on top of the 31 that fit
# before. Nine had to go, and the reasoning is the one already in this file
# -- randomise what MOVES and what is BIG -- carried one step further:
#
#   the two hip-roll servos (2)   bolted to the pelvis. They never move
#                                 relative to the root, so randomising them
#                                 and randomising the pelvis are very nearly
#                                 the same experiment -- the toe argument.
#   the four ankle servos (4)     pitch and roll, at the bottom of the leg,
#                                 near the ground and barely displaced by any
#                                 recovery. Both joints they act through keep
#                                 their damping draw.
#   the two ankle armatures (2)   already absent before B2 as the smallest of
#   ...and the two new ones       the eight at 5 kg*mm2. The two ankle-roll
#                                 armatures are the same 5 and go the same
#                                 way, which keeps the armature set exactly
#                                 the six biggest.
#   the two toes                  still absent, still welded to their feet.
#
# That is 31 again, one under the budget, and what remains is every link that
# carries real mass -- including the two new brackets -- every joint's
# damping, and the four servos that ride the swinging middle of the leg.
RANDOM = [assembly.randomise(item, "mass", scale=[0.9, 1.1])
          for item in [pelvis_c, hip_l_c, hip_r_c, thigh_l_c, thigh_r_c,
                       calf_l_c, calf_r_c, brk_l_c, brk_r_c, foot_l_c,
                       foot_r_c, sv_pitch_l_c, sv_pitch_r_c,
                       sv_knee_l_c, sv_knee_r_c]]
RANDOM = RANDOM + [assembly.randomise(item, "damping", scale=[0.8, 1.25])
                   for item in JOINTS]
RANDOM = RANDOM + [assembly.randomise(item, "armature", scale=[0.8, 1.25])
                   for item in JOINTS[:6]]

# ---------------------------------------------------------------------------
# The episode stops starting in the same place (M9, ADR-084).
#
# Every episode of the first run reset to the identical keyframe with every
# velocity zero and nothing ever pushed the machine. A posture found once was
# never tested, so "balance" was never the task -- which is the third reason
# bracing won, alongside the torque being available and the effort penalty
# being cheap.
#
# `reset_variation` moves the floating base RIGIDLY: a tilt whose direction
# is drawn over the whole circle, a lift, and a spin. It never touches joint
# angles, and that is the load-bearing decision -- the reset pose is the
# SOLVED configuration with both soles exactly on the floor, so a +-3 deg
# knee jitter moves a foot about 5 mm through it and MuJoCo answers that
# with an impulse nothing could stand up to. A rigid tilt cannot change the
# machine's shape, so it cannot self-interpenetrate however far it leans.
#
# The lift pays for the tilt, and the engine is what says how much.
#
# The first version of this asked for `height_mm=[0.0, 3.0]` against a 6 deg
# tilt, on the reasoning that 6 deg across a +-30 mm stance is about 3 mm at
# the sole. The engine refused it and printed the real number: **5.13 mm**,
# at an azimuth of 135 deg. The estimate was wrong because a tilt pivots
# about the base's own FRAME ORIGIN and the far thing from the pelvis origin
# is not the near sole edge -- it is a toe, diagonally, most of a leg away.
# That is exactly the arithmetic nobody should be doing by hand, and the
# reason the check measures at sixteen azimuths instead of documenting a
# formula.
#
# So the lift is 5.5-9.0 mm and the machine starts slightly airborne. A 9 mm
# drop lands at 0.42 m/s, which is itself a small disturbance and a welcome
# one: "arrive moving" is the same lesson as "arrive leaning".
#
# M9b (ADR-087) opens all three, and the LIFT is the one doing new work. A
# 15-45 mm drop lands at 0.54-0.94 m/s, and absorbing that is knee and ankle
# work that no shove is needed to demand -- it asks the machine to use all
# its joints before anything has pushed it. 15 deg of tilt needs about
# 49 mm x sin 15 deg = 12.7 mm of clearance by the near-sole estimate, and
# the 15 mm floor is meant to cover it; the engine's sixteen-azimuth check
# is what actually decides, and if it refuses, TAKE ITS NUMBER, exactly as
# the 5.13 mm correction above was taken. The estimate is wrong in a
# direction nobody can predict by hand, which is why the check exists.
#
# B1b adds the fourth knob and it is the one that changes what a batch is
# ABOUT. `capability.py` measured the m9c policy dying on the FIRST shove in
# every one of twelve episodes, at 0.51-3.93 s against a first shove landing
# at 1.02-1.96 s -- so the second shove window at 2.8-4.2 s had never once
# been reached, and the first second of every episode was spent standing
# still waiting to be pushed. A stumble IS an initial velocity, and injecting
# it at reset gives every episode a recovery to do from step 1.
#
# 250 mm/s is a capture point of about 32 mm on this machine -- past the
# 24.5 mm the sole reaches backward, inside the 45.5 it reaches forward. So
# the hard half of the draw starts each episode already needing a step and
# the easy half starts it needing an ankle, which is the same
# curriculum-inside-the-distribution the shove band uses.
#
# It is safe for the reason the rigid tilt is: a velocity cannot change the
# mechanism's shape, so it cannot drive a sole through the floor, and the
# clearance check has nothing to say about it. Its direction is drawn over
# the whole circle, like the tilt's, and it lands in the base's WORLD-frame
# linear velocity -- which is the other frame from the spin above it, in the
# same six numbers, and is MuJoCo's asymmetry rather than a choice here.
#
# B7 GIVES THE RESET THE TREATMENT THE SHOVE GOT: the drop opens downward,
# 15-45 mm -> 6.5-45 mm. The worst case is unchanged; what changes is that
# EASY EPISODES ARE IN THE BATCH FROM ITERATION 0. That is precisely the
# argument ADR-100 made for the shove band, and it was never applied to the
# reset -- so every episode of every run so far has begun with a 0.54 m/s
# landing to absorb before anything else could be attempted. With the shove
# going to 2.5 N there has to be somewhere in the distribution where the
# machine gets to just stand, and this is the cheapest place to put it.
# 6.5 mm lands at 0.36 m/s.
#
# 6.5 AND NOT 0, AND THE NUMBER IS THE ENGINE'S. B7 asked for [0.0, 45.0]
# and the sixteen-azimuth check refused the PAIR, exactly as it refused
# B1b's estimate above: `reset_variation_penetrates`, worst azimuth
# **225 deg**, **6.204 mm** of sole through the floor. The tilt is what
# needs the lift -- 15 deg swings a toe down most of a leg away from the
# pelvis origin -- so the floor cannot go below what the tilt costs, and
# hand arithmetic gets this wrong in a direction nobody can predict. 6.5 mm
# is that measurement with 0.3 mm over it, against a 0.1 mm tolerance.
#
# The alternative was narrowing `tilt_degrees`, and it was not taken: an
# arrival that leans is a recovery to do from step 1, which is the same
# thing the stumble velocity below buys and the reason B1b opened it.
start = assembly.reset_variation(pelvis_c,
                                 tilt_degrees=[0.0, 15.0],
                                 height_mm=[6.5, 45.0],
                                 angular_velocity_dps=[-90.0, 90.0],
                                 linear_velocity_mm_s=[0.0, 250.0],
                                 label="start")

# The forces, sized from the NEW limit rather than picked.
#
# One entry is one event. Two shoves and a wind, all applied at the pelvis
# centre of mass in the world frame (M9 phase 0 measured that this is where
# `xfrc_applied` acts; at the frame origin instead, every shove would carry
# a torque nobody declared).
#
# M9 sized these from the ankle: 0.35 N through a pelvis 145 mm up is 51
# N*mm of righting torque, inside the 86 N*mm the ankles have and inside the
# 117 N*mm the FOOT can spend before the sole rolls (ADR-082). That was the
# right ceiling for the question M9 asked -- "can it reject this IN PLACE" --
# and the run answered it: it stood, the hips moved, the knees and ankles did
# not, and the feet never left the floor.
#
# It never left the floor because nothing ever asked it to. The right measure
# is the CAPTURE POINT, xi = v / omega0 with omega0 = sqrt(g/h): the distance
# ahead of the feet where the centre of mass would have to be caught. This
# machine, measured off its own export, is
#
#     h = 146.0 mm  ->  omega0 = sqrt(9810/146.0) = 8.20 rad/s
#     m = 263.1 g
#     support polygon: 45.5 mm forward, 24.5 mm back, +-50 mm lateral
#
# and the M9 shove was 0.35 N x 0.12 s = 0.042 N*s -> 0.16 m/s -> a capture
# point of 19.5 mm. That is inside the polygon in EVERY direction including
# the narrow backward one. Nothing was asked of the knees because nothing
# needed to be. The sandbags are not in the feet; the sandbag is the shove.
#
# What the mechanism can actually reach, which is what bounds the new number:
# hip_pitch->knee 100 mm plus knee->ankle 95 mm is a 195 mm leg, and a 45 deg
# hip swing places a foot 138 mm from the hip -- so ONE step catches a
# capture point at 183 mm forward or 162 mm backward. The usable band is
# therefore 45-180 mm: past in-place recovery, inside single-step reach.
#
#     0.4 N over 0.12 s -> 0.048 N*s -> xi  22 mm   ankle, in place
#     0.8 N             -> 0.096      -> xi  45 mm   at the edge, hip strategy
#     1.4 N             -> 0.168      -> xi  78 mm   a step
#     2.0 N             -> 0.240      -> xi 111 mm   a definite step, catchable
#
# `newtons=[0.4, 2.0]` spanned that whole range in one draw, which is a
# curriculum INSIDE the distribution: an episode drawing 0.4 N is the old
# in-place problem and one drawing 2.0 N is the stepping problem, and the
# batch contains both from the first iteration. There is no scheduling
# feature in the surface and this deliberately does not need one.
#
# ---------------------------------------------------------------------------
# B3: THE BAND WAS OUT OF RANGE, AND THAT IS WHY NOTHING STOOD.
#
# Three runs produced nothing that survives a shove, and the reading was
# always about the training. `capability.py` measured the m9c policy against
# a swept fraction of this band and the reading changed completely:
#
#     no shove, reset variation on     11/12 survived, 556 steps of 600
#     x0.15  (0.06-0.30 N)             11/12            556
#     x0.30  (0.12-0.60 N)              7/12            446
#     x0.50  (0.20-1.00 N)              2/12            266
#     x1.00  (0.40-2.00 N, declared)    0/12            201
#
# It stands. It absorbs a 45 mm drop and a 15 degree lean, on 2-5 N*mm of
# mean torque against a limit of 86 -- balancing, not bracing, which is
# ADR-086's re-rating working. It then dies because it is asked to reject
# pushes three to six times beyond its mechanism's reach. The cliff is
# between a 0.30 N and a 0.60 N top end and the declared band starts at 0.40.
#
# So the band moves to what the machine can be trained to answer, and the
# capture-point arithmetic above is what sizes it (against the support
# polygon: 45.5 mm forward, 24.5 mm back, +-50 mm lateral):
#
#     0.15 N over 0.12 s -> xi   8 mm   inside the polygon everywhere
#     0.30 N             -> xi  17 mm   inside, including narrow backward
#     0.44 N             -> xi  24 mm   the backward edge
#     0.60 N             -> xi  33 mm   past backward, inside forward
#     0.90 N             -> xi  50 mm   past the forward edge: hip or a step
#
# `newtons=[0.15, 0.90]` keeps the curriculum idea and drops only the part of
# the range that was never answerable: half the draws land inside current
# capability, half past the edge, and the gradient runs through the middle.
# (The numbers above are computed at the OLD 146 mm centre of mass; B2 raised
# it to about 166, which lowers omega0 by 6 % and moves every xi up by the
# same, so they are a size rather than a boundary.)
#
# AIMED, which is B1a's whole purpose. `direction="horizontal"` draws the
# azimuth over the full circle, so most of every batch was spent in whatever
# direction the machine happened to be weakest in, unmeasured.
#
# WHICH DIRECTION THIS AIMS AT IS NOT WHAT IT WAS WRITTEN TO BE (ADR-107).
# `azimuth_degrees` is measured about **world +X**, and the engine has no
# concept of a machine's forward -- it cannot, and it never claimed to
# outside one wrong gloss in a docstring. THIS MACHINE FACES +Y: the hips sit
# at x = +-30, the toe geoms at y = +37.25, and the support polygon's 45.5 mm
# forward / 24.5 mm back run along Y. So `[-60, 60]` about +X is a LATERAL
# band, not the sagittal one this comment used to claim, and the first shove
# has been pushing across the machine all along.
#
# It was left exactly as it was, deliberately, until there was a reason to
# spend the digest. `azimuth_degrees` is a digest input: changing it changes
# the task bundle and `stand5.cxpolicy` stops loading against its own task.
# The comment that stood here said the number for the day it became worth it
# was `[30, 150]`, and B4 is that day.
#
# ---------------------------------------------------------------------------
# B4: NOTHING EVER ASKED IT TO STEP.
#
# `capability.py` on `stand5.cxpolicy`, twelve seeds, the sweep this project
# uses to tell a mechanism limit from a training result:
#
#     no shove, no reset variation     12/12 survived, 600 steps of 600
#     no shove, reset variation on     11/12            552
#     0.02-0.14 N  (x0.15)             11/12            552
#     0.04-0.27 N  (x0.30)             11/12            552
#     0.07-0.45 N  (x0.50)             11/12            552
#     0.11-0.68 N  (x0.75)              6/12            397
#     0.15-0.90 N  (x1.00, declared)    2/12            208   tipped 10
#
# It stands, it absorbs the stumble, and its cliff sits between a 0.45 N and
# a 0.68 N top end -- up from the 0.30-0.60 the m9c policy had, so the B3
# band did train something. It then dies TIPPED, ten times in twelve, and it
# never puts a foot out on the way down.
#
# Two reasons, and NEITHER OF THEM IS THE REWARD. Price a step under the
# weights ADR-087 arrived at -- 150 deg of joint travel, 30 deg of splay,
# 60 mm out, 800 mm/s -- and it comes to about -0.8 per step against the
# +1.00 `alive` pays, so a 150-step stumble that ends standing beats going
# down at once by roughly +400 over the episode. ADR-087 already did this
# arithmetic and it still holds. The reward pays for a recovery. The TASK
# never asked for one.
#
# FIRST: the aimed shove points the wrong way, which the comment above knew
# and deferred. `[-60, 60]` about world +X is LATERAL on a machine that
# faces +Y, so the one disturbance this task aims has been asking a sideways
# question for three runs. `[30, 150]` is the same +-60 deg arc about the
# machine's own forward.
#
# SECOND, AND THIS IS THE NEW FINDING: aimed forward, the declared band's
# TOP never demanded a step. The capture point, at this machine's measured
# 144.210 mm centre of mass and 302.01 g:
#
#     xi = v / omega0,  v = F x 0.12 s / 0.30201 kg,  omega0 = 8.248 rad/s
#        = F x 48.2 mm per newton
#
# against a support polygon reaching 45.5 mm forward and 24.5 mm back:
#
#     0.25 N -> xi 12 mm   inside the polygon in every direction: an ankle
#     0.51 N -> xi 25 mm   the BACKWARD edge
#     0.90 N -> xi 43 mm   two millimetres inside the forward edge  <- OLD TOP
#     0.94 N -> xi 46 mm   the FORWARD edge: the step boundary
#     1.20 N -> xi 58 mm   a quarter past it, and a third of the way to the
#                          183 mm that one step of this leg reaches
#
# The old top of 0.90 N lands just INSIDE the forward edge. Aimed forward it
# is precisely the hardest push a machine can answer without lifting a foot,
# so of course it never lifted one -- and aimed sideways, which is where it
# actually pointed, it is well past the +-50 mm lateral edge, which is why
# the deaths are all `tipped`. Either way the band never contained the
# problem the reward was rewritten to pay for.
#
# ---------------------------------------------------------------------------
# B5: AIM AT THE NARROW EDGE, NOT AT A BIGGER NUMBER (ADR-111).
#
# B4 aimed this shove forward and raised the band to 0.25-1.20 N so that its
# top would demand a step. 1500 iterations and 1.84 h later it had produced a
# WORSE machine, measured on the same task and the same twelve seeds:
#
#                          stand5      stand6 (B4)
#     survived              1/12         1/12
#     mean episode           166          144
#     episodes with a step   3/12         1/12
#     longest step         38.5 mm      14.8 mm
#     mean torque, hip roll  21/20        45/38
#     peak torque              81           85   (the limit is 86)
#     reward/step           0.027        0.126
#
# It learned to BRACE at maximum torque. Reward climbed, the cliff did not
# move, and the behaviour went backwards.
#
# THE MECHANISM, which is the whole finding. B4 argued that a band mostly
# past capability was still trainable because `alive` pays +1 every step, so
# the gradient does not need survival. That is exactly backwards. When most
# pushes are unanswerable, stiffening every joint RELIABLY buys extra steps
# of `alive`, and a step is a high-variance action that usually ends the
# episode SOONER. Attempting to step had negative expected value against
# bracing, so PPO correctly stopped attempting it. Dense survival reward does
# not make a hard task learnable; it makes stalling profitable.
#
# THE FIX IS NOT MORE FORCE. It is a narrower edge to be pushed over. The
# support polygon is 45.5 mm forward and only 24.5 mm BACK, and the capture
# point is F x 48.2 mm per newton, so:
#
#                   forward (45.5 mm)        backward (24.5 mm)
#     a step is     0.94 N and up            0.51 N and up
#     the cliff     ~0.6-0.9 N               ~0.4-0.6 N
#
# THE CLIFF IS DIRECTION-DEPENDENT, and it was measured rather than assumed:
# `capability.py` on stand5 against THIS aim gives 10/12 at a 0.40 N top and
# 3/12 at 0.60, where the same policy holds to 0.6-0.9 mixed. A narrow heel
# is harder to stand on, which is the same fact that makes it the right place
# to push. So the step boundary (0.51 N) and the cliff now COINCIDE instead
# of sitting 0.3 N apart -- which is the whole point, and is what B4 could
# not arrange from the front at any magnitude.
#
# WHY THAT BREAKS THE BRACE, precisely. It is not that the pushes are
# gentler; at the top of this band stand5 still dies (back 0/6). It is that
# bracing is no longer AVAILABLE as an answer. A 39 mm capture point outside
# a 24.5 mm heel cannot be held by stiffening at any torque, so for 58 % of
# the draws the choice is a foot or a fall -- where B4, aimed forward, let
# 73 % of its draws be answered by stiffening, and PPO took that deal. The
# 42 % below 0.51 N are still ankle work and still survivable, so the batch
# keeps the easy end the gradient comes from.
#
# `[210, 330]` is the same +-60 deg arc B4 used, about the machine's own
# BACKWARD (270 deg from world +X; it faces +Y, ADR-107). `[0.30, 0.80]`:
#
#     0.30 N -> xi 14 mm   inside the heel edge: an ankle answers it
#     0.51 N -> xi 25 mm   the BACKWARD edge -- the step boundary
#     0.65 N -> xi 31 mm   28 % past it
#     0.80 N -> xi 39 mm   57 % past it, and a fifteenth of the 162 mm one
#                          backward step of this leg reaches
#
# 58 % of draws now demand a step -- against B4's 27 % -- and the WHOLE band
# sits at or under the measured cliff rather than mostly past it. At the top
# of it bracing is not a losing strategy, it is an impossible one: 39 mm of
# capture point outside a 24.5 mm heel cannot be held by stiffening, so the
# only thing left that scores is a foot.
#
# The wind scales with the top as it always has: 0.09 -> 0.06.
#
# ---------------------------------------------------------------------------
# B4's own reasoning is kept below, because it is what this replaces.
#
# THE BAND IS PICKED AGAINST A MEASUREMENT, AND THE TWO RULES DISAGREE.
# `capability.py` was run again with the new AIM and the old policy, which is
# the honest "before" row for this run and is what sized the number:
#
#     0.04-0.20 N  (x0.15)   11/12      0.15-0.65 N  (x0.50)    8/12
#     0.09-0.39 N  (x0.30)   11/12      0.22-0.98 N  (x0.75)    1/12
#
# So aimed sagittally, stand5's cliff is a 0.39-0.65 N top end -- and the
# step boundary is 0.94 N. THE CLIFF SITS BELOW THE THING THE BAND EXISTS TO
# DEMAND. ADR-106's rule ("half the draws inside current capability") wants a
# top of 1.00 N; a top that actually requires a step wants 1.20 or more. They
# cannot both be had, and that gap IS the finding: this machine has never
# been trained anywhere near the strategy the task is now asking for.
#
# `[0.25, 1.20]` resolves it toward the demand and pays for it at the bottom:
# 42 % of draws are answerable by what the machine can already do, the top is
# a quarter past the forward edge and unmistakably a step, and the bottom
# comes DOWN rather than up so the easy end of the curriculum survives the
# top moving. ADR-106's mistake was a band 3-6x beyond reach with nothing
# answerable in it; this is 1.8x the cliff with two draws in five inside it.
#
# The gradient does not depend on survival, which is what makes 42 % enough:
# `alive` pays +1 every step, so an episode that lasts 300 steps and falls
# outscores one that lasts 100 and falls, and the batch is ranked long before
# anything stands all the way up.
#
# B7 TAKES IT PAST WHAT ANKLES CAN ANSWER: [0.30, 0.80] -> [0.40, 1.20] N.
#
# The arithmetic. xi is about 48 mm per newton on this machine, so the band
# spans xi 19 -> 58 mm against a 24.5 mm heel margin and a 45-180 mm
# single-step reach. B6's top was xi 39 mm -- a step the eye can barely see,
# and one an ankle can still very nearly cover. At 58 mm the capture point
# is more than twice the heel margin and a foot has to move.
#
# THE TOP WAS 2.50 N AND THE FIRST TWO PROBES SPENT THEMSELVES PROVING IT
# COULD NOT BE. Both collapsed the same way -- mean episode length falling
# monotonically to ~9 steps by iteration 150 where B6 bottomed at 20 and
# climbed to 137 -- and the reason is not that 2.5 N is beyond the
# mechanism. `feasibility.py` still says a step reaches it with 1.34-1.76x
# of margin, and `band_sweep.py` measured B6's own policy earning a POSITIVE
# +0.039 per step even at 2.5 N. The objective was fine. The DISTRIBUTION
# was not:
#
#     shove   reward/step   survived   (B6's policy, 12 seeds each)
#      0.4 N     +0.279      11/12
#      0.6 N     +0.164       8/12
#      0.8 N     +0.113       1/12
#      1.25 N    +0.039       0/12
#      2.5 N     +0.039       0/12
#
# THE WIDTH IS THE CURRICULUM AND WIDTH ALONE IS NOT ENOUGH -- what matters
# is what FRACTION of the batch is winnable, because the trainer has no
# schedule of any kind (ADR-100) and nothing else supplies a gradient.
# Uniform on [0.40, 2.50] puts only 19% of single draws inside B6's entire
# proven range, and with TWO shoves an episode it leaves about 4% of
# episodes winnable. B6's task was 100% winnable. Four per cent of a batch
# cannot carry the whole learning signal, and the other 96% is not merely
# uninformative: an early policy earns a negative per-step reward there and
# can END it by falling over, so those episodes actively train giving up.
#
# [0.40, 1.20] keeps the demand and restores the signal: the top is 1.5x
# B6's and unambiguously past what an ankle can answer, while about a third
# of draws land in the range B6 proved trainable. That is the same
# curriculum-inside-the-distribution argument ADR-100 made, with the
# fraction actually checked rather than assumed.
shove = assembly.disturbance(pelvis_c, newtons=[0.40, 1.20],
                             direction="horizontal",
                             azimuth_degrees=[210.0, 330.0],
                             at_seconds=[0.4, 1.6], duration_s=0.12,
                             label="shove")
# The second shove lands on an ALREADY-DISTURBED machine, and that is what it
# is for. The asymmetric states -- one foot forward, weight on one leg, hips
# out of square -- are exactly what M9 hazard 17 says not to reach by
# perturbing joint angles at reset, because a +-3 deg knee jitter puts a foot
# through the floor and MuJoCo answers with an impulse nothing survives. A
# machine mid-recovery is already in those states, legally, so the second
# push is how they get sampled without touching the reset pose.
#
# It ends by 4.32 s at the latest, leaving 1.68 s of the 6 s episode to
# recover in -- and a recovery is 1-2 s, so the window is deliberate.
#
# B3 moves both windows EARLIER and closer: 0.3-1.5 s and 1.8-3.6 s, where
# they were 0.8-2.0 and 2.8-4.2. Nothing on the old machine ever reached the
# second window, so half the episode's disturbance design had never run; and
# with a stumble at reset there is no longer any reason to leave the first
# second of every episode undisturbed. Earlier and denser is more recovery
# events per episode and less idle batch, which is the same currency the
# stumble buys in.
#
# THE FULL CIRCLE HERE, deliberately, where the first shove is aimed. Ankle
# roll was added so that a lateral push has an answer, and a task that never
# asks a lateral question would never find out whether it does. The azimuth
# split `compare.py` prints is what reads the answer back -- and since
# ADR-107 it reads it in the MACHINE's frame, so the column headed `lat`
# finally holds the lateral pushes.
#
# B4 moves its band with the first one and leaves its aim alone. The whole
# circle here is what keeps the lateral question being asked now that the
# first shove has stopped asking it by accident -- and a lateral push at the
# new top is 63 mm against a 50 mm lateral edge, so it wants a sideways step
# and the ankle roll B2 added is what has to answer.
#
# B7 MOVES THE WINDOWS APART AGAIN, and for the opposite reason B3 moved
# them together. B3's problem was that nothing ever reached the second
# window; B7's is that a 2.5 N recovery is a longer event than a 0.8 N one,
# and two shoves 0.2 s apart is one compound disturbance rather than two
# recoveries. 0.4-1.6 s and 2.4-4.0 s leaves EVERY draw at least 2.0 s of
# recovery room inside the 6 s horizon -- worst case a first shove ending at
# 1.72 s and a second starting at 2.4 s, and a second ending at 4.12 s with
# 1.88 s of episode after it.
#
# NO THIRD WINDOW, deliberately. Once the force is this big a third would
# rarely be reached at all, which is the Blocker-2 lesson from the m9 era:
# a disturbance nothing survives to see is batch spent on nothing.
shove2 = assembly.disturbance(pelvis_c, newtons=[0.40, 1.20],
                              direction="horizontal",
                              at_seconds=[2.4, 4.0], duration_s=0.12,
                              label="shove2")
# Wind is a push whose window is the whole episode -- the same mechanism,
# which is why there is no second surface for it. Something to lean against,
# not something to be knocked over by, and it scales with the shoves so it
# stays the same kind of thing: the band went 0.4-2.0 -> 0.15-0.9, so this
# goes 0.15 -> 0.07; B4's 0.25-1.20 carried it to 0.09 and B5's 0.30-0.80
# brings it back to 0.06. That is 9 N*mm at the ankle against the 86 it
# has: something to lean against, not something to be knocked over by.
#
# No arc on it. A wind that only ever blew from the front would be a
# constant the policy could learn to lean against once; drawing it over the
# whole circle keeps it a condition rather than a bias.
#
# B7 BREAKS THE SCALING and leaves this at 0.06. It has tracked the shove
# band every run so far, and carrying it to 2.5 N would put it at about
# 0.19 N -- 28 N*mm at the ankle against the 86 it has, sustained for the
# WHOLE episode. That is no longer something to lean against; it is a
# permanent lean, and it would be paid for by every one of the new terms at
# every step of every episode including the ones with no shove in them. The
# run is about a big transient, so the standing condition stays where it is.
wind = assembly.disturbance(pelvis_c, newtons=[0.0, 0.06],
                            direction="horizontal", sustained=True,
                            label="wind")

# 6 s at 100 Hz is 600 control steps, and 100 Hz divides the 0.002 s solver
# step exactly -- five steps to an action, with nothing rounded.
stand = assembly.task(model, actions=MOTORS, reward=REWARD,
                      termination=TERMINATION, randomisation=RANDOM,
                      reset_variation=[start],
                      disturbance=[shove, shove2, wind],
                      episode_seconds=6.0, control_hz=100, label="stand")

# ---------------------------------------------------------------------------
# The policy: B6 checkpoint 2400 of 2500 (ADR-112). 3.9 h on an RTX 5090,
# 2048 environments, seed 0, witness 2.82e-07.
#
# NOT `stand8.best.cxpolicy`, and the reason is the finding this run was
# built on. `best` tracks reward, and on this project reward selects the
# policy that does NOT step -- measured three times now. Across fifteen B6
# checkpoints at twelve seeds, `best` scores 1/12 on "stepped AND survived"
# where 2400 scores 6/12.
#
#     checkpoint 2400   survived 6/12   stepped 10/12   BOTH 6/12
#     stand8.best       survived 2/12   stepped  2/12   BOTH 1/12
#
# and against the policy it replaces, on `capability.py`'s scaled sweep:
#
#                      no shove   ~0.40 N   ~0.60 N   0.80 N
#     stand5             11/12     11/12      6/12       --
#     B6 checkpoint 2400 12/12     11/12     10/12     6/12
#
# Every death at every level is `tipped`; not one is `collapsed`, which is
# what says the crouch ADR-112 warned about never happened.
# RETIRED BY B7, AND NOT BY CHOICE. `stand8` was trained on 55 channels and
# this script declares 58 -- the `cam_*` triple ADR-116 added -- so the
# engine refuses it by name (`policy_observation_mismatch`) rather than
# feeding a 58-wide observation to a 55-wide first layer. That refusal is
# the point of declaring the channel list in the policy header, and there is
# no way to keep a policy across a channel change: the observation vector is
# positional.
#
# THE PROJECT THEREFORE HAS NO LIVE POLICY UNTIL B7'S CHECKPOINT LANDS, and
# `install_checkpoint.py` matches a policy call at the START OF A LINE, so
# reinstating this pair by hand is the first step of installing one. Both
# lines come back together: `play` is a rollout OF `balance`.
#
# balance = assembly.policy(stand, weights="stand8.cxpolicy",
#                           sha256="460296d1bcbbb27e2275a1788fd5796c55876559efda22fbd40e48990d91c842",
#                           label="balance")
# play = assembly.rollout(balance, frames_per_second=25, seed=7,
#                         label="stand_play")

# The policy, from 1 h 16 m on an RTX 5090: 2000 iterations, 4096
# environments, seed 0, reward/step -1.94 -> +0.391.
#
# The digest is required and never inferred. VISION principle 3 says any
# state that cannot be rebuilt from the script is a bug, and hours of
# stochastic GPU compute genuinely cannot be -- so the script carries the one
# thing that can be checked, which bytes it meant. On rebuild the worker
# re-checks the task bundle's digest, the model that bundle references, the
# observation channels in order, the action table, and re-evaluates the
# trainer's own witness with the engine's float64 forward pass.
#
# That last check is why ADR-081 exists: the previous machine's policy failed
# it at 1.43e-4 after 3 h 49 m, because `jax.vmap` had recorded the witness
# through a TF32 tensor core. This one agrees to 1.009e-07, 991x inside the
# tolerance, and the trainer now proves that before it writes the file.
# ...and `stand1.cxpolicy` is NOT that policy any more. Re-rating the eight
# motors from 216 N*mm to 86 changed the action table, which changed the
# model, which changed the task bundle's digest -- so the engine would
# refuse it, correctly and by name. Adding the reset variation and the three
# disturbances changed the bundle again.
#
# It is commented out rather than deleted because the digest is the record
# of what the braced policy was, and ADR-086's finding is a comparison
# against it. The next run's digest goes here.
#
# balance = assembly.policy(stand, weights="stand1.cxpolicy",
#                           sha256="6d99a5c78dcf6f20d1ae7adf47fe4164325078dd"
#                                  "1753bcf4299941e92afffb1c",
#                           label="balance")
# play = assembly.rollout(balance, frames_per_second=25, label="stand_play")

# The M9 policy, and it is commented out for the same reason `stand1` above
# is: the M9b reward, termination, observations, reset variation and forces
# all changed, so the task bundle's digest changed, so the engine would
# refuse this file by name -- correctly. The digest stays as the record of
# what the first *disturbed* policy was, because ADR-087's finding is a
# comparison against it.
#
# balance = assembly.policy(stand, weights="stand2.cxpolicy",
#                           sha256="74f9f127bc68cdb3c5024e9f3ed26fe0fafe1283"
#                                  "3c693a768f7019bed36d555d",
#                           label="balance")
# play = assembly.rollout(balance, frames_per_second=25, seed=7,
#                         label="stand_play")

# The M9b policy. `seed` is what makes this a *varied, disturbed* episode
# rather than the nominal one -- without it nothing is drawn and the rollout
# plays the solved pose with no forces on it, which is the very thing the
# slice exists to stop being the only question asked.
#
# It is iteration 100 of 500, and it is here to be WATCHED rather than
# because it works. The M9b run never recovered once -- 0 of 12 episodes at
# every one of its twenty checkpoints -- so there is no "best by survival" to
# install and this is not one. What it is, is the most alive the run ever
# got: 170 steps of 600 before it goes down, against 30 steps for the
# checkpoint the trainer labelled `best`.
#
# Two things are worth seeing in it, and neither is visible in the numbers.
# The legs MOVE -- peak hip and knee torque is 46-54 N*mm here against the
# 14-21 the same run commanded at iteration 25, which is the ADR-087 reward
# re-weighting doing exactly what it was written to do. And then it folds
# through the 87.6 mm `collapsed` floor anyway. "Uses its legs, and still
# goes down" is the state of this machine, and the shape of the fall is the
# evidence for what to fix next.
# ...and it is commented out for the same reason the two above it are, only
# more so. B2 put a tenth joint in the machine, so the observation vector is
# 52 channels where every policy in this project was trained on 46, and no
# amount of digest arithmetic would make one play: the engine refuses it BY
# NAME, which is the good failure. Nothing trained before B2 will ever run
# against this task again, and the digests stay as the record of what each
# run was.
#
# balance = assembly.policy(stand, weights="stand3.cxpolicy",
#                           sha256="1e60f1491cc3fb78dfcb16810432d4e1351550d"
#                                  "4944e4bf8d30ef3052bf7b237",
#                           label="balance")
# play = assembly.rollout(balance, frames_per_second=25, seed=7,
#                         label="stand_play")

# ---------------------------------------------------------------------------
# The B2 policy: 500 iterations, 4096 environments, 50 minutes on an RTX 5090.
#
# INSTALLED BY SURVIVAL, and this is iteration 250 of 500 rather than the end
# of the run. Twenty-one checkpoints played through the engine's reference
# runner over twelve seeds each: survival rose 0 -> 3 of 12 by iteration 125,
# held to 300, and then fell to 1 of 12 for the rest of the run while the
# trainer-s own reward stayed flat at about +0.19. Hazard 19-s rule is why
# this file names 250 and not `stand5.cxpolicy`.
#
# It ALSO happens to be the checkpoint the trainer labelled `best`, which is
# the first time in this project those two have agreed. That is a
# coincidence and not a reason: what chose it is `capability.py`, which
# sweeps a scale factor over the declared shove band and reported iteration
# 250 at or above iteration 150 on every row of the curve --
#
#     no shove, no reset variation   12/12      600 steps
#     no shove, variation on         11/12      554
#     x0.15  0.02-0.14 N             11/12      554
#     x0.30  0.04-0.27 N             11/12      554
#     x0.50  0.07-0.45 N             10/12      514
#     x0.75  0.11-0.68 N              7/12      431
#     x1.00  0.15-0.90 N (declared)   3/12      264
#
# -- where the single row at the declared band tied at 3/12 and could not
# tell them apart. One row is not a curve.
#
# WHAT THIS IS NOT. 3 of 12 at the declared band is not the majority Phase B
# set out to reach. The second half of that sentence used to read "`lat` is
# still the worst column, so the ankle rolls have not converted", and it was
# EXACTLY BACKWARDS (ADR-107): `compare.py` was quartering the circle about
# world +X on the assumption that +X was forward, and this machine faces +Y,
# so its `lat` column held the sagittal pushes. Re-measured in the machine's
# own frame at x0.50:
#
#     lateral   7/7      <- the strong axis
#     sagittal  3/5      <- the weak one
#
# The ankle roll servos B2 added ARE converting -- lateral is where this
# policy is perfect and sagittal is where it dies -- which is the opposite
# reading, off the same episodes, from one instrument being 90 degrees out.
# They are also being driven hard: 19 N*mm mean each, peaks to 67.
#
# It is worth being plain about what that costs. The mechanism change ADR-105
# made was argued for by the inverted reading. It was not thereby wrong --
# ankle roll and a longer pitch axis are load-bearing on any reading, and
# `feasibility.py` measured its margins in the mechanism's own frame -- but
# the sentence that motivated it named the wrong column.
#
# What did land: the band is in range (a graceful 12 -> 11 -> 10 -> 7 -> 3
# where m9c fell off a cliff), every death is `tipped` and not one is
# `collapsed`, and the second shove window is reached at last.
#
# `seed` is what makes this a VARIED, DISTURBED episode rather than the
# nominal one -- without it nothing is drawn and the rollout plays the solved
# pose with no forces on it, which is the very thing this slice exists to
# stop being the only question asked.
#
# SEED 6, and the choice is worth stating because a seed IS a choice. Three
# of twelve survive: 1, 6 and 11. Seed 1 survives having drawn a 0.29 N
# shove, which is barely a question; seed 6 draws 0.54 N at 1.14 s and
# 0.75 N at 1.85 s -- near the top of the declared band -- and stands the
# whole 600 steps. It is the hardest episode this policy is known to pass.
#
# WATCH IT FOR WHAT IT IS. Measured over that episode: the centre of mass
# falls 178 -> 145 mm, but that is the reset lift being absorbed rather than
# a stagger, and the deepest dip after the 0.75 N shove is 0.7 mm. Peak tilt
# is 0.056 against the 0.387 that terminates.
#
# THE FOOT DOES LIFT, and the sentence here used to say it never did. That
# came from a maximum taken over the WHOLE episode, which the 42 mm reset
# drop dominates: the foot starts at 42.7 mm, is on the floor by 1 s, and no
# later peak can beat where it began. Measured in the window after the drop
# is absorbed (t >= 1 s), against the floor it settles to:
#
#     left   8.69 -> 14.59 mm at 2.090 s     lift 5.91 mm
#     right  8.59 -> 12.72 mm at 2.110 s     lift 4.13 mm
#
# -- right after the 0.75 N shove at 1.85 s. A maximum over an interval that
# contains a much larger transient measures the transient, which is the same
# class of instrument error as ADR-107's rotated frame, found the same day.
#
# So this is a small weight shift with a real 6 mm lift in it, rather than
# the pure in-place ankle rejection it was first reported as -- and still not
# the visible loss-and-catch Phase B asked for. The episodes that get pushed
# harder do not stagger either: they tip.
#
# Change this to 7 to watch a representative FAILURE instead: 0.75 N at
# 1.85 s, tipped at step 247 of 600.
# ...and B4 retires it in turn, for the fourth time and the same reason: the
# aim and the band above are digest inputs, so this file is no longer the
# task `stand5` was trained on and the engine would refuse it by name. The
# digest stays as the record of what the best standing policy was before the
# task asked for a step, because B4's finding is a comparison against it:
#
#     stand5, 500 iterations, 50.7 min, 4096 envs, seed 0
#     reward/step 0.185 at the end, best 0.203 at iteration 299 of 500
#     episode 269.5 steps of 600
#
# balance = assembly.policy(stand, weights="stand5.cxpolicy",
#                           sha256="5ee3d6863a69c34a929933004ede62008457a2b"
#                                  "a9f67f5ce77cf2309ecd5b93a",
#                           label="balance")
# play = assembly.rollout(balance, frames_per_second=25, seed=6,
#                         label="stand_play")

result = {
    "pelvis_solid": pelvis_solid,
    "floor_solid": floor_solid,
    "hip_left": hip_l,
    "hip_right": hip_r,
    "thigh_left": thigh_l,
    "thigh_right": thigh_r,
    "calf_left": calf_l,
    "calf_right": calf_r,
    "ankle_bracket_left": brk_l,
    "ankle_bracket_right": brk_r,
    "foot_left": foot_l,
    "foot_right": foot_r,
    "toe_left": toe_l,
    "toe_right": toe_r,
    "sv_roll_left": sv_roll_l,
    "sv_roll_right": sv_roll_r,
    "sv_pitch_left": sv_pitch_l,
    "sv_pitch_right": sv_pitch_r,
    "sv_knee_left": sv_knee_l,
    "sv_knee_right": sv_knee_r,
    "sv_ankle_left": sv_ankle_l,
    "sv_ankle_right": sv_ankle_r,
    "sv_aroll_left": sv_aroll_l,
    "sv_aroll_right": sv_aroll_r,
    "ground": ground,
    "pelvis": pelvis_c,
    "hip_l": hip_l_c,
    "hip_r": hip_r_c,
    "thigh_l": thigh_l_c,
    "thigh_r": thigh_r_c,
    "calf_l": calf_l_c,
    "calf_r": calf_r_c,
    "ankle_bracket_l": brk_l_c,
    "ankle_bracket_r": brk_r_c,
    "foot_l": foot_l_c,
    "foot_r": foot_r_c,
    "toe_l": toe_l_c,
    "toe_r": toe_r_c,
    "sv_roll_l": sv_roll_l_c,
    "sv_roll_r": sv_roll_r_c,
    "sv_pitch_l": sv_pitch_l_c,
    "sv_pitch_r": sv_pitch_r_c,
    "sv_knee_l": sv_knee_l_c,
    "sv_knee_r": sv_knee_r_c,
    "sv_ankle_l": sv_ankle_l_c,
    "sv_ankle_r": sv_ankle_r_c,
    "sv_aroll_l": sv_aroll_l_c,
    "sv_aroll_r": sv_aroll_r_c,
    "hip_roll_l": hip_roll_l,
    "hip_roll_r": hip_roll_r,
    "hip_pitch_l": hip_pitch_l,
    "hip_pitch_r": hip_pitch_r,
    "knee_l": knee_l,
    "knee_r": knee_r,
    "ankle_l": ankle_l,
    "ankle_r": ankle_r,
    "ankle_roll_l": ankle_roll_l,
    "ankle_roll_r": ankle_roll_r,
    "toe_l_joint": toe_l_j,
    "toe_r_joint": toe_r_j,
    "sv_roll_l_joint": sv_roll_l_j,
    "sv_roll_r_joint": sv_roll_r_j,
    "sv_pitch_l_joint": sv_pitch_l_j,
    "sv_pitch_r_joint": sv_pitch_r_j,
    "sv_knee_l_joint": sv_knee_l_j,
    "sv_knee_r_joint": sv_knee_r_j,
    "sv_ankle_l_joint": sv_ankle_l_j,
    "sv_ankle_r_joint": sv_ankle_r_j,
    "sv_aroll_l_joint": sv_aroll_l_j,
    "sv_aroll_r_joint": sv_aroll_r_j,
    "asm": asm,
    "diag": diag,
    "model": model,
    "stand": stand,
    # "balance": balance,
    # "stand_play": play,
    # The one output the shell can PLAY. A script that declares a task and no
    # rollout builds, publishes and animates nothing -- which is exactly what
    # this file did between B2 and B6, and it looks from the app like the
    # simulation broke rather than like it was never asked for.
    #
    # It is what this file does again, for as long as B7 is training: the
    # channel count moved, so no policy that exists can be declared here.
    # Both rows come back with the two calls above them.
}
