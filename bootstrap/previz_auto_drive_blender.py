"""auto_drive.py — Blender previz blockout for Wan VACE.
A hatchback blockout drives a gentle S-curve road; the camera tracks at 3/4 and slowly orbits (-35°..+30°) with a
slight rise. Vertical 480x832, 81 frames @16 fps (Wan native 5 s). Two passes: flat-colour (Workbench, object colours)
and depth (near = white) for the VACE control video. Usage: blender -b --python auto_drive.py -- <out_dir> [pass=both|color|depth]"""
import bpy, math, sys, os, random
from mathutils import Vector
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else '/tmp/previz'; PASS = argv[1] if len(argv) > 1 else 'both'
os.makedirs(OUT, exist_ok=True); random.seed(7)
FR, FPS, W, H = 81, 16, 480, 832
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene; sc.frame_start, sc.frame_end = 1, FR; sc.render.fps = FPS
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
def mat(name, rgb):
    m = bpy.data.materials.new(name); m.diffuse_color = (*rgb, 1); return m
def obj_color(o, rgb): o.color = (*rgb, 1)
# --- road: S-curve bezier, extruded ribbon 7 m wide, plus dashed centre line as separate thin ribbon
cu = bpy.data.curves.new('road_path', 'CURVE'); cu.dimensions = '3D'; sp = cu.splines.new('BEZIER')
pts = [(-60, -8, 0), (-25, 6, 0), (5, -6, 0), (35, 7, 0), (70, -4, 0)]
sp.bezier_points.add(len(pts) - 1)
for bp, p in zip(sp.bezier_points, pts):
    bp.co = p; bp.handle_left_type = bp.handle_right_type = 'AUTO'
path = bpy.data.objects.new('road_path', cu); sc.collection.objects.link(path)
cu.path_duration = FR; cu.use_path = True
road_cu = cu.copy(); road_cu.extrude = 0; road_cu.bevel_depth = 0
road = bpy.data.objects.new('road', road_cu); sc.collection.objects.link(road)
road_cu.dimensions = '2D'; road_cu.fill_mode = 'BOTH'
# ribbon via a flat profile curve used as bevel object
prof = bpy.data.curves.new('road_prof', 'CURVE'); ps = prof.splines.new('POLY'); ps.points.add(1)
ps.points[0].co = (-3.5, 0, 0, 1); ps.points[1].co = (3.5, 0, 0, 1)
prof_o = bpy.data.objects.new('road_prof', prof); sc.collection.objects.link(prof_o); prof_o.hide_render = True
road_cu.dimensions = '3D'; road_cu.bevel_mode = 'OBJECT'; road_cu.bevel_object = prof_o
road.location.z = 0.02; obj_color(road, (0.16, 0.16, 0.18))
line_cu = cu.copy(); line = bpy.data.objects.new('centre_line', line_cu); sc.collection.objects.link(line)
lp = bpy.data.curves.new('line_prof', 'CURVE'); ls = lp.splines.new('POLY'); ls.points.add(1)
ls.points[0].co = (-0.12, 0, 0, 1); ls.points[1].co = (0.12, 0, 0, 1)
lp_o = bpy.data.objects.new('line_prof', lp); sc.collection.objects.link(lp_o); lp_o.hide_render = True
line_cu.bevel_mode = 'OBJECT'; line_cu.bevel_object = lp_o; line.location.z = 0.04; obj_color(line, (0.92, 0.9, 0.8))
# ground
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, 0)); g = bpy.context.object; obj_color(g, (0.42, 0.47, 0.36))
# --- car blockout (hatchback proportions, metres): body 4.1 x 1.8, cabin set back, 4 wheels
def box(name, size, loc, rgb, bevel=0.12):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc); o = bpy.context.object; o.name = name; o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    m = o.modifiers.new('bev', 'BEVEL'); m.width = bevel; m.segments = 3; obj_color(o, rgb); return o
car = bpy.data.objects.new('car', None); sc.collection.objects.link(car)
body = box('body', (4.1, 1.8, 0.62), (0, 0, 0.62), (0.72, 0.74, 0.78), 0.18)
cabin = box('cabin', (2.3, 1.62, 0.62), (-0.35, 0, 1.22), (0.25, 0.3, 0.36), 0.22)
# taper the cabin: pull the top verts inward (windscreen / rear-window slope)
for v in cabin.data.vertices:
    if v.co.z > 0: v.co.x *= 0.72; v.co.y *= 0.9
for o in (body, cabin): o.parent = car
wheels = []
for x in (1.35, -1.35):
    for y in (0.86, -0.86):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.34, depth=0.26, vertices=24, location=(x, y, 0.34), rotation=(math.pi / 2, 0, 0))
        w = bpy.context.object; obj_color(w, (0.05, 0.05, 0.05)); w.parent = car; wheels.append(w)
for x in (2.02,):   # headlights
    for y in (0.6, -0.6):
        l = box('lamp', (0.08, 0.34, 0.14), (x, y, 0.78), (1, 0.95, 0.75), 0.02); l.parent = car
# follow the road: Follow Path on the car root, offset driven by frame (constant speed)
fp = car.constraints.new('FOLLOW_PATH'); fp.target = path; fp.use_curve_follow = True; fp.forward_axis = 'FORWARD_X'
fp.use_fixed_location = True
fp.offset_factor = 0.18; fp.keyframe_insert('offset_factor', frame=1)
fp.offset_factor = 0.52; fp.keyframe_insert('offset_factor', frame=FR)
for fc in car.animation_data.action.fcurves:
    for k in fc.keyframe_points: k.interpolation = 'LINEAR'
# wheel spin from travelled distance: path length * 0.34 of the curve over FR frames
L = sum((Vector(pts[i + 1]) - Vector(pts[i])).length for i in range(len(pts) - 1)) * 0.34
turns = L / (2 * math.pi * 0.34)
for w in wheels:
    w.rotation_mode = 'XYZ'; w.keyframe_insert('rotation_euler', frame=1)
    w.rotation_euler.y += turns * 2 * math.pi; w.keyframe_insert('rotation_euler', frame=FR)
    for fc in w.animation_data.action.fcurves:
        for k in fc.keyframe_points: k.interpolation = 'LINEAR'
# --- environment: sample the REAL road centreline (evaluated curve -> mesh) and keep scenery 9+ m away from it
dg = bpy.context.evaluated_depsgraph_get()
tmp = bpy.data.meshes.new_from_object(path.evaluated_get(dg))
centre = [path.matrix_world @ v.co for v in tmp.vertices]
def clear_of_road(p, m):
    return min(((p.x - c.x) ** 2 + (p.y - c.y) ** 2) for c in centre) >= m * m
placed = 0
for k in range(0, len(centre) - 1, 3):
    c, n = centre[k], centre[k + 1]; d = (n - c).normalized(); side_v = Vector((-d.y, d.x, 0))
    for side in (1, -1):
        if random.random() < 0.4: continue
        off = random.uniform(11, 18); p = c + side_v * side * off
        if not clear_of_road(p, 10): continue
        hgt = random.uniform(4, 16)
        box('bld', (random.uniform(3, 6), random.uniform(3, 6), hgt), (p.x, p.y, hgt / 2),
            random.choice([(0.78, 0.74, 0.68), (0.6, 0.64, 0.7), (0.85, 0.82, 0.74), (0.55, 0.5, 0.46)]), 0.05); placed += 1
    if random.random() < 0.6:
        side = random.choice((1, -1)); p = c + side_v * side * random.uniform(7.5, 9.5)
        if clear_of_road(p, 7):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=1.6, location=(p.x, p.y, 0.8)); obj_color(bpy.context.object, (0.35, 0.25, 0.15))
            bpy.ops.mesh.primitive_ico_sphere_add(radius=1.2, subdivisions=2, location=(p.x, p.y, 2.3)); obj_color(bpy.context.object, (0.2, 0.42, 0.22))
print('scenery placed', placed)
for i in range(9):
    bpy.ops.mesh.primitive_cone_add(radius1=random.uniform(25, 45), depth=random.uniform(18, 35), vertices=7,
                                    location=(-80 + i * 22, random.choice((1, -1)) * random.uniform(80, 110), 8))
    obj_color(bpy.context.object, (0.5, 0.55, 0.62))
# --- camera rig: pivot follows the car position (copy location), orbits -35..+30 deg, slight rise
pivot = bpy.data.objects.new('cam_pivot', None); sc.collection.objects.link(pivot)
cl = pivot.constraints.new('COPY_LOCATION'); cl.target = car
cr = pivot.constraints.new('COPY_ROTATION'); cr.target = car; cr.use_x = cr.use_y = False
cam_d = bpy.data.cameras.new('cam'); cam_d.lens = 40; cam_d.sensor_fit = 'VERTICAL'
cam = bpy.data.objects.new('cam', cam_d); sc.collection.objects.link(cam); sc.camera = cam; cam.parent = pivot
orb = bpy.data.objects.new('orbit', None); sc.collection.objects.link(orb); orb.parent = pivot; cam.parent = orb
cam.location = (12.5, 0, 3.6)
tt = cam.constraints.new('TRACK_TO'); tt.target = body; tt.track_axis = 'TRACK_NEGATIVE_Z'; tt.up_axis = 'UP_Y'
orb.rotation_euler.z = math.radians(-40); orb.keyframe_insert('rotation_euler', frame=1)
orb.rotation_euler.z = math.radians(25); orb.keyframe_insert('rotation_euler', frame=FR)
cam.keyframe_insert('location', frame=1); cam.location = (11.0, 0, 5.2); cam.keyframe_insert('location', frame=FR)
for o in (orb, cam):
    for fc in o.animation_data.action.fcurves:
        for k in fc.keyframe_points: k.interpolation = 'BEZIER'; k.easing = 'EASE_IN_OUT'
# --- render settings
sc.render.engine = 'BLENDER_WORKBENCH'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'OBJECT'; sh.show_cavity = True; sh.show_shadows = True
sc.world = bpy.data.worlds.new('w'); sc.world.color = (0.72, 0.8, 0.9)
sc.render.image_settings.file_format = 'PNG'
def render(name):
    sc.render.filepath = os.path.join(OUT, name, 'f_'); bpy.ops.render.render(animation=True)
if PASS in ('both', 'color'):
    sc.use_nodes = False; render('color')
if PASS in ('both', 'depth'):
    bpy.context.view_layer.use_pass_z = True; sc.use_nodes = True; nt = sc.node_tree; nt.nodes.clear()
    rl = nt.nodes.new('CompositorNodeRLayers'); nz = nt.nodes.new('CompositorNodeMapRange'); nz.inputs['From Min'].default_value = 6; nz.inputs['From Max'].default_value = 40; nz.inputs['To Min'].default_value = 1; nz.inputs['To Max'].default_value = 0; nz.use_clamp = True
    comp = nt.nodes.new('CompositorNodeComposite')
    nt.links.new(rl.outputs['Depth'], nz.inputs['Value']); nt.links.new(nz.outputs[0], comp.inputs['Image'])
    render('depth')
print('PREVIZ_DONE', OUT)
