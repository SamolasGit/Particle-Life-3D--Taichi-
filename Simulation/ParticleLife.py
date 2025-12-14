import taichi as ti
import taichi.math as tm
import random

ti.init(arch=ti.gpu)

NUM_PARTICLES = 1
SPECIES = 8
SPACE_SIZE = 200.0
SPEED = 0.0004
RADIUS = 20.0
BETA = 8.0
TIME_STEP = 1

CAMERA_SPEED = 0.6

REACTION_DISTANCE = 0.5
reaction_count = ti.field(ti.i32, ())
reaction_count[None] = 0

color_list = [
    tm.vec3(1.0, 0.0, 0.0),
    tm.vec3(0.0, 0.0, 1.0),
    tm.vec3(0.0, 1.0, 0.0),
    tm.vec3(1.0, 1.0, 0.0),
    tm.vec3(1.0, 0.0, 1.0),
    tm.vec3(0.0, 1.0, 1.0),
    tm.vec3(0.5, 0.5, 0.5),
    tm.vec3(1.0, 0.5, 0.0),

    tm.vec3(0.5, 0.0, 1.0),
    tm.vec3(0.7, 0.2, 0.2),
    tm.vec3(0.2, 0.6, 0.2),
    tm.vec3(0.2, 0.4, 0.8),
    tm.vec3(0.9, 0.3, 0.6),
    tm.vec3(0.9, 0.8, 0.4),
    tm.vec3(0.3, 0.9, 0.8),
    tm.vec3(0.6, 0.9, 0.2),
    tm.vec3(0.9, 0.6, 0.9),
    tm.vec3(0.4, 0.3, 0.2),
    tm.vec3(0.8, 0.8, 1.0),
    tm.vec3(1.0, 0.8, 0.8),

    tm.vec3(0.1, 0.1, 0.6),
    tm.vec3(0.6, 0.1, 0.1),
    tm.vec3(0.1, 0.6, 0.6),
    tm.vec3(0.6, 0.6, 0.1),
    tm.vec3(0.7, 0.3, 0.7),
    tm.vec3(0.3, 0.7, 0.7),
    tm.vec3(0.7, 0.7, 0.3),
    tm.vec3(0.2, 0.2, 0.2),
    tm.vec3(0.9, 0.9, 0.9),
    tm.vec3(0.3, 0.1, 0.5),

    tm.vec3(0.5, 0.1, 0.3),
    tm.vec3(0.1, 0.5, 0.3),
    tm.vec3(0.3, 0.5, 0.1),
    tm.vec3(0.1, 0.3, 0.5),
    tm.vec3(0.5, 0.3, 0.1),
    tm.vec3(0.3, 0.1, 0.7),
    tm.vec3(0.1, 0.7, 0.1),
    tm.vec3(0.7, 0.1, 0.3),
    tm.vec3(0.1, 0.3, 0.7),
    tm.vec3(0.7, 0.3, 0.1),

    tm.vec3(0.4, 0.0, 0.4),
    tm.vec3(0.0, 0.4, 0.4),
    tm.vec3(0.4, 0.4, 0.0),
    tm.vec3(0.8, 0.4, 0.6),
    tm.vec3(0.6, 0.8, 0.4),
    tm.vec3(0.4, 0.6, 0.8),
    tm.vec3(0.2, 0.8, 0.4),
    tm.vec3(0.8, 0.2, 0.4),
    tm.vec3(0.4, 0.2, 0.8),
    tm.vec3(0.2, 0.4, 0.8)
]

pos = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
vel = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
species = ti.field(ti.i32, NUM_PARTICLES)
color = ti.Vector.field(3, ti.f32, NUM_PARTICLES)
species_colors = ti.Vector.field(3, ti.f32, SPECIES)
attract = ti.field(ti.f32, (SPECIES, SPECIES))

reaction_table = ti.Vector.field(2, ti.i32, (SPECIES, SPECIES))

MAX_COLORS = len(color_list)
taichi_color_list = ti.Vector.field(3, ti.f32, MAX_COLORS)
for k in range(MAX_COLORS):
    taichi_color_list[k] = color_list[k]

@ti.kernel
def setup_species_colors_kernel():
    for i in range(SPECIES):
        species_colors[i] = taichi_color_list[i % MAX_COLORS]


@ti.kernel
def initialize_particles_kernel(space_size: ti.f32):
    for i in range(NUM_PARTICLES):
        pos[i] = tm.vec3(
            (ti.random() - 0.5) * space_size,
            (ti.random() - 0.5) * space_size,
            (ti.random() - 0.5) * space_size
        )
        vel[i] = tm.vec3(0, 0, 0)
        s = ti.random(ti.i32) % SPECIES
        species[i] = s
        color[i] = species_colors[s]


@ti.func
def force_func(r: ti.f32, a: ti.f32, radius: ti.f32, beta: ti.f32) -> ti.f32:
    f = 0.0
    if r < radius:
        if r < beta:
            f = r / beta - 1.0
        else:
            f = a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta))
    return f


@ti.kernel
def update_particles_kernel(space_size: ti.f32, radius: ti.f32, beta: ti.f32, time_step: ti.f32):
    half_space = space_size * 0.5
    for i in range(NUM_PARTICLES):
        s = species[i]
        p = pos[i]
        force = tm.vec3(0.0, 0.0, 0.0)

        for j in range(NUM_PARTICLES):
            if i != j:
                d = pos[j] - p
                dist = d.norm() + 1e-5

                # ---------- REACTION CHECK ----------
                if dist < REACTION_DISTANCE:
                    a = species[i]
                    b = species[j]
                    cd = reaction_table[a, b]

                    if cd[0] != -1:
                        species[i] = cd[0]
                        species[j] = cd[1]

                        color[i] = species_colors[cd[0]]
                        color[j] = species_colors[cd[1]]

                        reaction_count[None] += 1

                f = force_func(dist, attract[s, species[j]], radius, beta)
                force += d.normalized() * f

        vel[i] += force * SPEED * time_step
        vel[i] *= 0.98
        pos[i] += vel[i] * time_step

        for k in ti.static(range(3)):
            if pos[i][k] > half_space:
                pos[i][k] -= space_size
            elif pos[i][k] < -half_space:
                pos[i][k] += space_size


def randomize_reactions(max_reactions):
    used_pairs = set()
    count = 0

    while count < max_reactions:
        a = random.randint(0, SPECIES - 1)
        b = random.randint(0, SPECIES - 1)

        if (a, b) in used_pairs:
            continue

        c = random.randint(0, SPECIES - 1)
        d = random.randint(0, SPECIES - 1)
        if c == a and d == b:
            continue

        used_pairs.add((a, b))
        set_reaction(a, b, c, d)
        print(f"Reaction: {a}+{b} -> {c}+{d}")
        count += 1


@ti.kernel
def init_reaction_table():
    for i in range(SPECIES):
        for j in range(SPECIES):
            reaction_table[i, j] = ti.Vector([-1, -1])


def set_reaction(a, b, c, d):
    reaction_table[a, b] = ti.Vector([c, d])


def randomize_attraction():
    for i in range(SPECIES):
        for j in range(SPECIES):
            attract[i, j] = random.uniform(-0.1, 0.1)


@ti.kernel
def print_defined_reactions():
    for i in range(SPECIES):
        for j in range(SPECIES):
            c = reaction_table[i, j]
            if c[0] != -1:
                print(i, "+", j, " -> ", c[0], "+", c[1])


setup_species_colors_kernel()
randomize_attraction()
initialize_particles_kernel(SPACE_SIZE)
init_reaction_table()
randomize_reactions(10)
print_defined_reactions()

window = ti.ui.Window("Particle Life 3D", (1920, 1080))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(0, 0, 80)
paused = False

line = ti.Vector.field(3, dtype=ti.f32, shape=2)

while window.running:
    camera.track_user_inputs(window, movement_speed=CAMERA_SPEED, hold_key=ti.ui.RMB)

    SPACE_SIZE = window.get_gui().slider_float("SPACE", SPACE_SIZE, 0, 1000)
    BETA = window.get_gui().slider_float("BETA", BETA, 0, 20)
    RADIUS = window.get_gui().slider_float("RADIUS", RADIUS, 0, 40)
    TIME_STEP = window.get_gui().slider_float("TIME STEP", TIME_STEP, 0, 2)

    if window.get_event(ti.ui.PRESS):
        key = window.event.key
        if key == ti.ui.SPACE:
            paused = not paused
        CAMERA_SPEED = 1 if window.is_pressed(ti.ui.SHIFT) else 0.6

    if not paused:
        update_particles_kernel(SPACE_SIZE, RADIUS, BETA, TIME_STEP)

    scene.set_camera(camera)
    cam_pos = camera.curr_position
    scene.point_light(pos=cam_pos, color=(1, 1, 1))

    half = SPACE_SIZE * 0.5

    v = [
        (-half, -half, -half),
        ( half, -half, -half),
        ( half,  half, -half),
        (-half,  half, -half),
        (-half, -half,  half),
        ( half, -half,  half),
        ( half,  half,  half),
        (-half,  half,  half)
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for a, b in edges:
        line[0] = v[a]
        line[1] = v[b]
        scene.lines(line, width=2, color=(1, 1, 1))

    scene.particles(pos, per_vertex_color=color, radius=0.3)

    canvas.scene(scene)
    window.show()
