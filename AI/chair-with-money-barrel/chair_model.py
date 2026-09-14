"""
Chair with a banknote barrel + friction-fit lid with a bow.
Units: millimetres. Target printer: Prusa MK4S (250 x 210 x 220 mm).
"""
import numpy as np
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import mapbox_earcut
from shapely import affinity
from shapely.geometry import Polygon, Point, LineString, box as sbox
from shapely.ops import unary_union
from trimesh.creation import box, cylinder, revolve, torus, icosphere
from trimesh.transformations import rotation_matrix as R, translation_matrix as T

U = lambda parts: trimesh.boolean.union(parts, engine="manifold")
D = lambda a, b: trimesh.boolean.difference([a, b], engine="manifold")

# ---------------------------------------------------------------- parameters
SEAT_W, SEAT_D = 108.0, 100.0      # seat cushion
SEAT_Z0, SEAT_Z1 = 66.0, 80.0     # cushion bottom / top above the floor
BACK_W, BACK_T = 88.0, 12.0       # back panel width / thickness


# 2000 Kc note = 164 x 74 mm. Rolled along its long edge it becomes a 74 mm
# tall tube; bore D34 (circumference 107 mm) means ~1.5 turns, still a loose
# roll. 92 mm of clear depth = 18 mm of headroom over the note.
BARREL_R_OUT = 21.4               # outer radius (D42.8), 4.4 mm wall
BORE_R = 17.0                     # inner radius (D34)
BARREL_TOP = 170.0                # absolute Z of barrel rim
BORE_BOTTOM = SEAT_Z1 - 2.0       # bore sinks 2 mm into the seat
CLEAR = 0.18                      # radial clearance: friction fit, not a slip fit

SKIRT_IR = BARREL_R_OUT + CLEAR   # 27.35 - cap slides over the barrel
SKIRT_WALL = 2.8
RIB_STAND = 0.20                  # friction ribs inside the cap skirt
RIB_N = 6
SKIRT_OR = SKIRT_IR + SKIRT_WALL  # 30.35
SKIRT_H = 13.0                    # how deep the cap swallows the barrel
ROOF_T = 4.0
CAP_TOP = SKIRT_H + ROOF_T        # 21 - top face of the cap

NAME = "$"                     # embossed on the barrel
NAME_SIZE = 16.0                  # cap height-ish, mm
NAME_DEPTH = 1.3                  # how far the letters stand proud
NAME_Z = 118.0                    # baseline height above the floor

RIB_T = 1.9                       # ribbon thickness of the bow
BOW_S = 0.74                      # bow scale


def brick(sx, sy, sz, cx=0.0, cy=0.0, cz=0.0):
    return box(extents=(sx, sy, sz), transform=T([cx, cy, cz]))




# ------------------------------------------------------------------- helpers
def rounded_rect(w, h, r):
    return sbox(-w / 2 + r, -h / 2 + r, w / 2 - r, h / 2 - r).buffer(r, quad_segs=8)


def loft(poly, levels):
    """Stack scaled copies of one outline and skin them: tapers and chamfers."""
    ring = np.asarray(poly.exterior.coords)[:-1]
    n = len(ring)
    verts = np.vstack([np.column_stack([ring * sc, np.full(n, z)]) for sc, z in levels])

    faces = []
    for i in range(len(levels) - 1):
        a, b = i * n, (i + 1) * n
        for k in range(n):
            k2 = (k + 1) % n
            faces += [[a + k, a + k2, b + k2], [a + k, b + k2, b + k]]
    cap = mapbox_earcut.triangulate_float64(ring, np.array([n])).reshape(-1, 3)
    faces += list(cap[:, ::-1])                                    # bottom
    faces += list(cap + (len(levels) - 1) * n)                     # top

    m = trimesh.Trimesh(verts, np.array(faces), process=True)
    m.fix_normals()
    return m


def smooth(ctrl, n=160):
    """Centripetal Catmull-Rom through 2D control points."""
    p = np.asarray(ctrl, float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        P = p[i - 1:i + 3]
        d = np.linalg.norm(np.diff(P, axis=0), axis=1) ** 0.5
        t = np.concatenate([[0], np.cumsum(np.maximum(d, 1e-6))])
        tt = np.linspace(t[1], t[2], max(4, n // (len(p) - 3)), endpoint=False)
        A = [(t[j + 1] - tt)[:, None] / (t[j + 1] - t[j]) * P[j]
             + (tt - t[j])[:, None] / (t[j + 1] - t[j]) * P[j + 1] for j in range(3)]
        B = [(t[j + 2] - tt)[:, None] / (t[j + 2] - t[j]) * A[j]
             + (tt - t[j])[:, None] / (t[j + 2] - t[j]) * A[j + 1] for j in range(2)]
        out.append((t[2] - tt)[:, None] / (t[2] - t[1]) * B[0]
                   + (tt - t[1])[:, None] / (t[2] - t[1]) * B[1])
    return np.vstack(out + [p[-2][None, :]])


def sweep(section, path, scales=None, up=(0.0, 0.0, 1.0)):
    """Run a 2D cross-section along any 3D path, using parallel transport so
    the section never spins. Section columns are (across, up)."""
    path = np.asarray(path, float)
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]

    n = np.asarray(up, float) - np.dot(up, t[0]) * t[0]
    if np.linalg.norm(n) < 1e-6:
        n = np.array([1.0, 0.0, 0.0]) - t[0][0] * t[0]
    normals = [n / np.linalg.norm(n)]
    for i in range(1, len(path)):                    # propagate, then re-orthogonalise
        n = normals[-1] - np.dot(normals[-1], t[i]) * t[i]
        normals.append(n / np.linalg.norm(n))
    n = np.array(normals)
    u = np.cross(n, t)
    u /= np.linalg.norm(u, axis=1)[:, None]

    sec = np.asarray(section.exterior.coords)[:-1]
    m, M = len(sec), len(path)
    if scales is None:
        scales = np.ones((M, 2))
    scales = np.asarray(scales, float)
    if scales.ndim == 1:
        scales = np.column_stack([scales, scales])

    verts = np.concatenate([path[i] + np.outer(sec[:, 0] * scales[i, 0], u[i])
                                    + np.outer(sec[:, 1] * scales[i, 1], n[i])
                            for i in range(M)])
    faces = []
    for i in range(M - 1):
        a, b = i * m, (i + 1) * m
        for k in range(m):
            k2 = (k + 1) % m
            faces += [[a + k, a + k2, b + k2], [a + k, b + k2, b + k]]
    cap = mapbox_earcut.triangulate_float64(sec, np.array([m])).reshape(-1, 3)
    faces += list(cap[:, ::-1]) + list(cap + (M - 1) * m)

    mesh = trimesh.Trimesh(verts, np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def path3d(ctrl, n=140):
    """Smooth 3D control points into a swept path."""
    return smooth(ctrl, n)


def round_end(path, d=9.0, u_min=0.90, v_min=0.22):
    """Scale factors that round off the free end of a swept panel."""
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    x = np.clip((seg[-1] - seg) / d, 0, 1)
    f = np.sqrt(np.clip(1 - (1 - x) ** 2, 0, 1))
    return u_min + (1 - u_min) * f, v_min + (1 - v_min) * f


def barrel_radius(z):
    return float(np.interp(z, [p[1] for p in BARREL_PROFILE],
                              [p[0] for p in BARREL_PROFILE]))


# ------------------------------------------------------------ embossed name
def text_polygons(txt, size):
    """Glyph outlines of `txt` as a shapely (Multi)Polygon, baseline at y=0."""
    font = FontProperties(fname=__import__("matplotlib.font_manager", fromlist=["x"])
                          .findfont(FontProperties(family="DejaVu Sans", weight="bold")))
    rings = TextPath((0, 0), txt, size=size, prop=font).to_polygons(closed_only=True)
    polys = [Polygon(r) for r in rings if len(r) > 2]
    polys = [p.buffer(0) for p in polys if p.is_valid or True]
    polys.sort(key=lambda p: p.area, reverse=True)

    solids, holes = [], []
    for p in polys:
        if any(s.contains(p.representative_point()) for s in solids) and \
           not any(h.contains(p.representative_point()) for h in holes):
            holes.append(p)
        else:
            solids.append(p)
    shape = unary_union(solids)
    for h in holes:
        shape = shape.difference(h)
    return shape


def wrapped_text(txt, radius, z_base, theta0=-np.pi / 2):
    """Extruded text bent around a cylinder of the given radius."""
    shape = text_polygons(txt, NAME_SIZE)
    geoms = list(getattr(shape, "geoms", [shape]))
    flat = trimesh.util.concatenate(
        [trimesh.creation.extrude_polygon(g, height=NAME_DEPTH + 1.0) for g in geoms])
    flat = flat.subdivide_to_size(1.0)             # so the bend stays smooth

    v = flat.vertices.copy()
    minx, maxx = v[:, 0].min(), v[:, 0].max()
    v[:, 0] -= (minx + maxx) / 2                   # centre the string

    theta = theta0 + v[:, 0] / radius              # reads left->right from outside
    r = radius - 1.0 + v[:, 2]                     # 1 mm buried in the wall
    out = np.column_stack([r * np.cos(theta), r * np.sin(theta), z_base + v[:, 1]])
    flat.vertices = out
    flat.fix_normals()
    return flat


# ------------------------------------------------------------------ the chair
# modern canister barrel: near-straight taper, straight neck for the cap
BARREL_PROFILE = [(BARREL_R_OUT + 2.4, SEAT_Z1 - 3.0),     # soft foot flare
                  (BARREL_R_OUT, SEAT_Z1 + 5.5),
                  (BARREL_R_OUT, BARREL_TOP - 1.2),
                  (BARREL_R_OUT - 1.2, BARREL_TOP)]          # lead-in for the cap


def barrel_radius(z):
    return float(np.interp(z, [p[1] for p in BARREL_PROFILE],
                              [p[0] for p in BARREL_PROFILE]))


def circle(r, seg=28):
    return Point(0, 0).buffer(r, quad_segs=seg)


def rod(r_top, r_foot, length, lean, direction, base, waist=0.86):
    """Tapered round leg / spindle, leaning `lean` degrees towards `direction`."""
    m = loft(circle(r_top), [(r_foot / r_top, 0.0),
                             (waist, length * 0.55),
                             (1.0, length)])
    m.apply_translation([0, 0, -length])                     # pivot at the top
    d = np.radians(lean)
    m.apply_transform(R(-d * direction[1], [1, 0, 0]) @ R(d * direction[0], [0, 1, 0]))
    m.apply_translation([base[0], base[1], base[2]])
    return m


# --------------------------------------------------------------- the chair
# Modern shell chair: one moulded shell, seat flowing into the back, carried
# on four slim splayed legs.
SHELL_W = 106.0                   # widest point of the shell
SHELL_T = 6.6                     # shell thickness
SHELL_CUP = 8.0                   # how far the shell curls up at the sides
SEAT_Z1 = 80.0                    # top of the seat at the centre

SHELL_CTRL = [(-46.0, 80.5), (-30.0, 77.2), (-8.0, 76.4), (14.0, 76.9),
              (30.0, 79.8), (42.0, 86.5), (49.0, 99.0), (54.0, 118.0),
              (57.0, 140.0), (58.0, 157.0)]

BARREL_PROFILE = [(BARREL_R_OUT + 2.4, SEAT_Z1 - 4.0),     # soft foot flare
                  (BARREL_R_OUT, SEAT_Z1 + 5.5),
                  (BARREL_R_OUT, BARREL_TOP - 1.2),
                  (BARREL_R_OUT, BARREL_TOP - 1.2),
                  (BARREL_R_OUT - 1.2, BARREL_TOP)]        # lead-in for the cap


def barrel_radius(z):
    return float(np.interp(z, [p[1] for p in BARREL_PROFILE],
                              [p[0] for p in BARREL_PROFILE]))


def circle(r, seg=28):
    return Point(0, 0).buffer(r, quad_segs=seg)


def turned(ctrl, radii, seg=12, n=90):
    """A tapered round leg: smooth path plus a radius profile along it."""
    p = path3d(ctrl, n)
    r = np.interp(np.linspace(0, 1, len(p)), np.linspace(0, 1, len(radii)), radii)
    return sweep(circle(1.0, seg), p, r)


def shell_section(width, cup, thickness):
    """Constant-thickness dished band - the shell's cross-section."""
    u = np.linspace(-width / 2, width / 2, 41)
    v = cup * (2 * u / width) ** 2
    return LineString(np.column_stack([u, v])).buffer(thickness / 2, quad_segs=8)


def make_chair():
    parts = []

    # --- the shell: swept side profile, dished across its width -----------
    sp = path3d([(0.0, y, z) for y, z in SHELL_CTRL], n=170)
    f = np.linspace(0, 1, len(sp))
    width = np.interp(f, [0, 0.34, 0.56, 0.80, 1.0],
                         [SHELL_W, 100.0, 94.0, 100.0, 104.0]) / SHELL_W
    ea, eb = round_end(sp, d=6.0, u_min=0.985, v_min=0.45)
    fa, fb = round_end(sp[::-1], d=6.0, u_min=0.985, v_min=0.45)
    parts.append(sweep(shell_section(SHELL_W, SHELL_CUP, SHELL_T), sp,
                       np.column_stack([width * ea * fa[::-1], eb * fb[::-1]])))

    # --- four slim legs, splayed out front and back ------------------------
    for sx in (-1, 1):
        parts.append(turned([(sx * 47.0, -44.0, 0.0),
                             (sx * 42.0, -37.0, 40.0),
                             (sx * 37.0, -30.0, SEAT_Z1)], [4.6, 5.8, 6.9]))
        parts.append(turned([(sx * 45.0, 41.0, 0.0),
                             (sx * 40.0, 32.0, 40.0),
                             (sx * 35.0, 24.0, SEAT_Z1)], [4.6, 5.8, 6.9]))

    chair = U(parts)

    # --- barrel --------------------------------------------------------------
    barrel = revolve([[0, BARREL_PROFILE[0][1]]] + [list(p) for p in BARREL_PROFILE]
                     + [[0, BARREL_TOP]], sections=96)
    chair = U([chair, barrel])

    bore = revolve([[0, BORE_BOTTOM],
                    [BORE_R, BORE_BOTTOM],
                    [BORE_R, BARREL_TOP - 1.2],
                    [BORE_R + 1.2, BARREL_TOP + 0.5],
                    [0, BARREL_TOP + 0.5]], sections=96)
    chair = D(chair, bore)
    chair = D(chair, brick(400, 400, 100, cz=-50.0))      # flat feet

    name = wrapped_text(NAME, barrel_radius(NAME_Z + NAME_SIZE * 0.35), NAME_Z)
    return U([chair, name])


def ribbon(path, widths, thick):
    """Loft a flat ribbon of given widths along a 3D path (band normal = +Z)."""
    path = np.asarray(path, float)
    widths = np.asarray(widths, float)
    n = np.array([0.0, 0.0, 1.0])

    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    side = np.cross(tang, n)
    side /= np.linalg.norm(side, axis=1)[:, None]

    hw = (widths / 2)[:, None]
    ht = thick / 2
    rings = np.stack([path + side * hw + n * ht,
                      path - side * hw + n * ht,
                      path - side * hw - n * ht,
                      path + side * hw - n * ht], axis=1)   # (N, 4, 3)

    N = len(path)
    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(N - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            faces += [[a + k, a + k2, b + k2], [a + k, b + k2, b + k]]
    faces += [[0, 1, 2], [0, 2, 3]]                                  # start cap
    a = 4 * (N - 1)
    faces += [[a + 2, a + 1, a], [a + 3, a + 2, a]]                  # end cap

    m = trimesh.Trimesh(verts, np.array(faces), process=True)
    m.fix_normals()
    return m


def teardrop_loop(length, bulge, w_far, w_knot, curl, sections=120):
    """Pinched ribbon loop: leaves the knot, bulges out, returns to the knot."""
    t = np.linspace(0.18, 2 * np.pi - 0.18, sections)
    f = (1 - np.cos(t)) / 2                      # 0 at the knot, 1 at the far end
    pts = np.column_stack([length * f,
                           bulge * np.sin(t) * f,
                           curl * f ** 1.5])
    w = w_knot + (w_far - w_knot) * f ** 0.6
    return ribbon(pts, w, RIB_T)


def tail(length, width, z_knot, z_rest, splay, sections=56):
    """Ribbon end that leaves the knot and lies flat on the cap's top face."""
    t = np.linspace(0, 1, sections)
    pts = np.column_stack([splay * length * t ** 2,
                           length * t,
                           z_rest + (z_knot - z_rest) * (1 - t) ** 2.4])
    band = ribbon(pts, width * (1 - 0.12 * t), RIB_T)
    notch = box(extents=(width, width, 6 * RIB_T),
                transform=T(pts[-1] + [0, width * 0.66, 0]) @ R(np.radians(45), [0, 0, 1]))
    return D(band, notch)


def make_bow(top_z):
    parts = []

    # two pinched loops: built pointing +X, tilted up, mirrored, splayed
    for s in (-1, 1):
        loop = teardrop_loop(length=30.0, bulge=13.5, w_far=9.5, w_knot=4.5, curl=3.0)
        loop.apply_transform(R(np.radians(-19.0), [0, 1, 0]))        # far end lifts
        if s < 0:
            loop.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
            loop.fix_normals()
        loop.apply_transform(R(np.radians(s * 15.0), [0, 0, 1]))     # splay
        loop.apply_translation([0, 0, 3.6])
        parts.append(loop)

    # two ribbon ends falling forward over the rim
    for s, ln in ((-1, 30.0), (1, 26.0)):
        tl = tail(length=ln, width=8.5, z_knot=4.6, z_rest=0.45, splay=0.10)
        tl.apply_transform(R(np.radians(180 + s * 24.0), [0, 0, 1]))
        
        parts.append(tl)

    # knot: soft pillow with a ribbon belt wrapped around it
    knot = icosphere(subdivisions=3, radius=6.2)
    knot.apply_transform(np.diag([1.30, 1.0, 0.82, 1.0]))
    knot.apply_translation([0, 0, 3.6])
    parts.append(knot)

    belt = revolve([[4.4, 0], [7.0, 0], [7.4, 0.8], [7.4, 5.2],
                    [7.0, 6.0], [4.4, 6.0], [4.4, 0]], sections=64)
    belt.apply_transform(R(np.radians(90), [0, 1, 0]) @ T([0, 0, -3.0]))
    belt.apply_translation([0, 0, 3.6])
    parts.append(belt)

    bow = U(parts)
    bow.apply_transform(np.diag([BOW_S, BOW_S, BOW_S, 1.0]))
    bow.apply_translation([0, 0, top_z])

    # the ribbon must not pass through the cap: anything below the cap's top
    # face has to be either on the axis (the knot, buried in the roof) or
    # completely clear of the cap's outer wall.
    v = bow.vertices
    r = np.hypot(v[:, 0], v[:, 1])
    bad = (v[:, 2] < top_z - 0.8) & (r > 12.0) & (r < SKIRT_OR + 0.5)
    assert not bad.any(), f"{int(bad.sum())} bow vertices cut through the cap wall"
    return bow


# -------------------------------------------------------------------- the lid
def make_lid():
    """Cap that drops over the barrel rim. Printed as-is: the skirt rim sits on
    the bed and the roof bridges the cavity, which is a hidden inside surface."""
    body = revolve([[0, SKIRT_H],
                    [SKIRT_IR, SKIRT_H],            # roof underside
                    [SKIRT_IR, 1.2],                # inner wall
                    [SKIRT_IR + 1.2, 0],            # lead-in chamfer
                    [SKIRT_OR, 0],                  # bottom rim
                    [SKIRT_OR, CAP_TOP - 1.5],      # outer wall
                    [SKIRT_OR - 1.5, CAP_TOP],      # top chamfer
                    [0, CAP_TOP],
                    [0, SKIRT_H]], sections=128)
    lid = U([body, make_bow(CAP_TOP)])

    # hard keep-out: the barrel sweeps up this cylinder, so no ribbon may live
    # in it. Guarantees the cap seats on the rim whatever the bow does.
    keepout = cylinder(radius=SKIRT_IR, sections=128,
                       segment=[[0, 0, -60.0], [0, 0, SKIRT_H]])
    lid = D(lid, keepout)

    # friction ribs: six short lines of contact instead of one big cylinder.
    # Far more forgiving than a tight bore - if the cap binds, sand the ribs.
    ribs = []
    rr = 1.1
    for k in range(RIB_N):
        a = 2 * np.pi * k / RIB_N
        dc = SKIRT_IR - RIB_STAND + rr
        ribs.append(cylinder(radius=rr, sections=16,
                             segment=[[dc * np.cos(a), dc * np.sin(a), 1.8],
                                      [dc * np.cos(a), dc * np.sin(a), SKIRT_H - 1.0]]))
    return U([lid] + ribs)


if __name__ == "__main__":
    chair = make_chair()
    lid = make_lid()

    for name, m in (("chair", chair), ("lid", lid)):
        print(f"{name:6s} watertight={m.is_watertight} volume={m.volume/1000:8.1f} cm3 "
              f"bbox={np.round(m.extents,1)}")

    chair.export("/mnt/user-data/outputs/chair_with_barrel.stl")
    lid.export("/mnt/user-data/outputs/lid_with_bow.stl")
