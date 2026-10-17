import numpy as np
import moderngl
import pygame
from pygame import display
from pyrr import Matrix44
from V2 import *






class Planet:
    def __init__(self, radius, color, center=(0,0,0),name = "None",sun=None,rin=16,seg=32,grav_min_dist=15):
        self.name = name
        self.center = np.array(center, dtype='f4')
        self.radius = radius
        self.color = color
        self.vertices,self.indices = generate_indexed_uv_sphere(radius, rings=16, segments=32, center=center)
        self.rotation_angle = 0.0
        self.rotation_speed =  random.uniform(10.0, 70.0)
        self.orbit_speed = [-1,1][random.randint(0,1)] * ((random.randint(2,20)/1000)*self.rotation_speed) 
        self.orbit_inclination = random.randint(-14, 14)/10  
        self.real_center = self.center
        if sun != None:
            self.orbit_radius  =  np.linalg.norm(self.center-sun)
        else:
            self.orbit_radius  = 0
        self.gravity_min_dist = grav_min_dist
        self.vbo = ctx.buffer(self.vertices.tobytes())
        self.vao = ctx.simple_vertex_array(prog, self.vbo, 'in_position')


class Solar_Sys:
    def __init__(self, lst_planet):
        self.lst_planet = lst_planet


#-------------------creation of stars --------------------------
num_planets_by_sys = 10
lst_solar_sys = []
radius_star = 2500000//10000
stars_base = generate_sphere_points(100,radius_star)
for point in stars_base:
    lst_planet_tmp = []
    points = [point[0]+random.randint(-radius_star//5,radius_star//5),point[1]+random.randint(-radius_star//5,radius_star//5),point[2]+random.randint(-radius_star//5,radius_star//5)]
    sun_size = random.randint(100,400)/1000
    lst_planet_tmp.append(Planet(sun_size, np.array([1.0, 1.0, 0.0]),points))
    for i in range(2,random.randint(5,num_planets_by_sys)):
        lst_planet_tmp.append(Planet(radius=random.randint(2,5)/15, color=np.random.uniform(0.2, 1.0, size=3).astype('f4'),center=(points[0]+i*(sun_size*8),points[1],points[2]),sun=(points[0],points[1],points[2]),grav_min_dist=3,rin=128,seg=128))
    lst_solar_sys.append(Solar_Sys(lst_planet_tmp))      


# solar sys us 
Sun = Planet(2/10, (1.0, 1.0, 0.0),name="Sun",center =(200/100,0,0))
Earth = Planet(2/100, (0.2, 0.5, 1.0), center=( 350/100,0,0),name="Earth",sun=(Sun.center[0],Sun.center[1],Sun.center[2]),grav_min_dist=3)
Mars = Planet(1.5/100, (0.9, 0.1, 0.2), center=( 500/100,0,0),name="Marth",sun=(Sun.center[0],Sun.center[1],Sun.center[2]),grav_min_dist=3)
solar_sys = Solar_Sys([Sun, Earth,Mars])
lst_solar_sys.append(solar_sys)


# ------------------ instanciating of planet ( stars too ) --------------
# stuff GPU for Planet 
vertices, indices = generate_indexed_uv_sphere(radius=1.0, rings=16, segments=32)
vbo_all = ctx.buffer(vertices.tobytes())
ibo_all = ctx.buffer(indices.tobytes())
instance_data = []
for sys_solar in lst_solar_sys:
    for planet in sys_solar.lst_planet:
        instance_data.extend([
            *planet.center, 
            planet.rotation_angle, 
            planet.radius, 
            *planet.color, 
            planet.rotation_speed,
            planet.orbit_radius,
            planet.orbit_speed,
            planet.orbit_inclination
        ])

max_instances = 400000  
vbo_instance = ctx.buffer(reserve=max_instances*12*4) 
vbo_instance.write(np.array(instance_data, dtype='f4').tobytes())
# vao instance
vao_all = ctx.vertex_array(
    prog,
    [
        (vbo_all, '3f', 'in_position'),
        (vbo_instance, '3f 1f 1f 3f 1f 1f 1f 1f/i', 
         'instance_center', 'instance_angle', 'instance_scale', 'instance_color', 'instance_rotation_speed','instance_orbit_radius',
         'instance_orbit_speed','instance_orbit_inclination')
    ],
    ibo_all  # faces
)


Cam = Camera()

# STUFF GPU for orbit circle in 3D
circle_vertices = generate_circle_points(128)
vbo_circle = ctx.buffer(circle_vertices.tobytes())
max_orbit_distances = 500
instance_circle_data = []
for sys_solar in lst_solar_sys:
    sun = sys_solar.lst_planet[0]

    for planet in sys_solar.lst_planet[1:]:  
        
        instance_circle_data.extend([
            *sun.center,       
            planet.orbit_radius,      
            0.8, 0.8, 0.8,
            planet.orbit_inclination     
        ])
vbo_circle_instance = ctx.buffer(np.array(instance_circle_data, dtype='f4').tobytes())
vao_circle = ctx.vertex_array(
    circle_prog,  
    [
        (vbo_circle, '3f', 'in_position'),
        (vbo_circle_instance, '3f 1f 3f 1f/i', 'instance_center', 'instance_scale', 'instance_color', 'instance_orbit_inclination')
    ]
)





stop_it = True
key_press = {}

running = True
while running:
    dt = clock.tick(60) / 1000.0  
    t = get_time()*Cam.t
    time_uniform.value = pygame.time.get_ticks() / 1000.0  # secondes
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key_press[event.key] = True
        elif event.type == pygame.KEYUP:
            key_press[event.key] = False
        elif event.type == pygame.VIDEORESIZE:
            width, height = event.size
            pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
            ctx.viewport = (0, 0, width, height)
            Cam.aspect_ratio = width / height






    #CAM stuff for proj
    Cam.UpdateForProj()


    
    closest_sys = min(lst_solar_sys, key=lambda p: np.linalg.norm(Cam.camera_pos - p.lst_planet[0].real_center)- p.lst_planet[0].radius)
    
    closest_planet = min(closest_sys.lst_planet[1:], key=lambda x: np.linalg.norm(Cam.camera_pos - x.real_center)- x.radius)
    

    # Cam Focus 
    if np.linalg.norm(Cam.camera_pos - closest_planet.real_center)- closest_planet.radius < closest_planet.gravity_min_dist :
        if Cam.is_on_orbit == False:
            Cam.orbit_planet_last_pos =  closest_planet.real_center
            Cam.move_speed = 0.5
            Cam.is_on_orbit = True
            stop_it = False

        Cam.rot_speed = 50
        
        planet_to_cam = Cam.camera_pos - closest_planet.real_center
        planet_to_cam /= np.linalg.norm(planet_to_cam)  
        Cam.up = planet_to_cam
        Cam.forward = Cam.up
                     
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)  
        reference /= np.linalg.norm(reference)

        Cam.forward = Cam.up - np.dot(Cam.up, reference) * reference
        Cam.forward /= np.linalg.norm(Cam.forward)
        Cam.camera_pos += closest_planet.real_center-Cam.orbit_planet_last_pos-Cam.up*0.00001

        Cam.orbit_planet_last_pos =  closest_planet.real_center
                


    elif np.linalg.norm(Cam.camera_pos - closest_planet.real_center)- closest_planet.radius > closest_planet.gravity_min_dist:
        # remettre les axes par défaut
    
        if stop_it == False:
            Cam.up / np.linalg.norm(Cam.up)
            pitch_rad = np.arcsin(-Cam.up[1])
            Cam.pitch = np.degrees(pitch_rad)
            if abs(Cam.up[1]) < 0.999:
                
                yaw_rad = np.arcsin(-Cam.up[0]/np.cos(pitch_rad))
                Cam.yaw = np.degrees(yaw_rad)
                if Cam.up[2] > 0:
                    Cam.yaw = 180.0 - Cam.yaw
                #("go", np.degrees(pitch_rad),np.degrees(yaw_rad),Cam.up,np.cos(pitch_rad) * np.cos(yaw_rad))
                stop_it = True
        Cam.is_on_orbit = False
        Cam.rot_speed = 60.0
        Cam.up = np.array([0.0, 1.0, 0.0], dtype='f4')
        # rotation
        Cam.HandleRotation(key_press,dt)

    #Movement
    Cam.HandleMovement(key_press,dt)




    # SYS PLAN MOVEMNT STUFF 
    いとうかんあこ  = 3000.0  
    instance_data = []

    for sys_solar in lst_solar_sys:
          
       
        sun = sys_solar.lst_planet[0]
        instance_data.extend([*sun.center, sun.rotation_angle,sun.radius, *sun.color,sun.rotation_speed,0,0,0])
    
    sun = closest_sys.lst_planet[0]
    for planet in closest_sys.lst_planet[1:]:
    
        instance_data.extend([*sun.center, planet.rotation_angle,planet.radius, *planet.color,planet.rotation_speed,planet.orbit_radius,planet.orbit_speed,planet.orbit_inclination])
        planet.real_center = get_shader_center(planet,t,sun)

    
    prog['time'].value = t
    # update buffer
    
    vbo_instance.write(np.array(instance_data, dtype='f4').tobytes())
    projection_uniform.write(Cam.projection.astype('f4').tobytes())
    view_uniform.write(Cam.view.astype('f4').tobytes())


    # special proj for circle because they need another shader and vao vbo they are not the same as planet and also another projection bcs of the render distance being different and not handle the same ways as the little planet above in the code 
    circle_prog['projection'].write(Cam.projection_circle.astype('f4').tobytes())
    circle_prog['view'].write(Cam.view.astype('f4').tobytes())
    

    #BG
    ctx.clear(0.0, 0.0, 0.0, depth=1.0)  # fond sombre
    vao_all.render(mode=moderngl.TRIANGLES, instances= len(instance_data) // 12)
    vao_circle.render(mode=moderngl.LINE_LOOP, instances=len(instance_circle_data) // 7)
    screen.fill((0,255,0))

    
    display.flip()
