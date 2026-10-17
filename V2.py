import numpy as np
import moderngl
import pygame
import random
import math
from pygame import display
from pyrr import Matrix44
import time


pygame.init()

start_time = time.perf_counter()
pygame.display.set_caption("SolarSys0.1.3")
screen = display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE | pygame.GL_MULTISAMPLEBUFFERS)
clock = pygame.time.Clock()
ctx = moderngl.create_context()
ctx.enable(moderngl.DEPTH_TEST)
ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA




# --- SHADERS ---




vertex_shader = '''
#version 330

uniform mat4 projection;
uniform mat4 view;
uniform float time;

in vec3 in_position;

// instance attributes
in vec3 instance_center;              // center sun
in float instance_angle;              // rotation propre init
in float instance_scale;              // radius
in vec3 instance_color;
in float instance_rotation_speed;     // spin
in float instance_orbit_radius;       // dist sun
in float instance_orbit_speed;        // speed orbitale
in float instance_orbit_inclination;  // inclination orbital

out vec3 v_color;
out vec3 v_planet_center;             // center  with t

void main() {

    // ORBITE 
    float orbit_angle = time * instance_orbit_speed;

    mat3 orbit_rot = mat3(
        cos(orbit_angle), 0.0, sin(orbit_angle),
        0.0,              1.0, 0.0,
       -sin(orbit_angle), 0.0, cos(orbit_angle)
    );

    mat3 orbit_inclined_rot = mat3(
        1.0, 0.0, 0.0,
        0.0, cos(instance_orbit_inclination), -sin(instance_orbit_inclination),
        0.0, sin(instance_orbit_inclination),  cos(instance_orbit_inclination)
    );

    vec3 orbit_offset =
        orbit_inclined_rot *
        orbit_rot *
        vec3(instance_orbit_radius, 0.0, 0.0);

    // CENTRE RÉEL DE LA PLANÈTE
    v_planet_center = instance_center + orbit_offset;

    //  ROTATION PROPRE
    float spin_angle = radians(instance_angle)
                     + instance_rotation_speed * time;

    mat3 spin_rot = mat3(
        cos(spin_angle), 0.0, sin(spin_angle),
        0.0,             1.0, 0.0,
       -sin(spin_angle), 0.0, cos(spin_angle)
    );

    vec3 local_pos = spin_rot * (in_position * instance_scale);

    // POSITION FINALE
    vec3 world_pos = v_planet_center + local_pos;

    gl_Position = projection * view * vec4(world_pos, 1.0);
    v_color = instance_color;
}
'''
# color stuff 
fragment_shader = '''
#version 330
in vec3 v_color;
out vec4 f_color;


void main() {
    f_color = vec4(v_color,1.0);
}
'''

vertex_shader_circle = '''
#version 330
uniform mat4 projection;
uniform mat4 view;

in vec3 in_position;
in vec3 instance_center;
in float instance_scale;
in vec3 instance_color;
in float instance_orbit_inclination; 
out vec3 v_color;

void main() {
    // matrice de rotation autour de l'axe x pour l'inclinaison
    mat3 orbit_inclined_rot = mat3(
        1, 0, 0,
        0, cos(instance_orbit_inclination), -sin(instance_orbit_inclination),
        0, sin(instance_orbit_inclination),  cos(instance_orbit_inclination)
    );

    vec3 world_pos = instance_center + orbit_inclined_rot * (in_position * instance_scale);
    gl_Position = projection * view * vec4(world_pos, 1.0);
    v_color = instance_color;
}
'''

fragment_shader_cricle = '''
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color,0.3);
}
'''

prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

circle_prog =  ctx.program(vertex_shader=vertex_shader_circle, fragment_shader=fragment_shader_cricle)


projection_uniform = prog['projection']
view_uniform = prog['view']
time_uniform = prog['time']



def rotate_around_axis(v, axis, angle):
    axis = axis / np.linalg.norm(axis)  # assure que l’axe est unitaire
    c = np.cos(angle)
    s = np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)

def distance_squared(a, b):
    return np.sum((a - b)**2)



def get_time():
    return time.perf_counter() - start_time


def get_shader_center(planet, time, sun):
    # Calcul de l'angle orbitale (comme dans le shader)
    orbit_angle = time * planet.orbit_speed
    orbit_rot = np.array([
        [np.cos(orbit_angle), 0, np.sin(orbit_angle)],
        [0, 1, 0],
        [-np.sin(orbit_angle), 0, np.cos(orbit_angle)]
    ], dtype='f4')

    # Inclinaison orbitale
    incl_angle = planet.orbit_inclination
    orbit_inclined_rot = np.array([
        [1, 0, 0],
        [0, np.cos(incl_angle), -np.sin(incl_angle)],
        [0, np.sin(incl_angle),  np.cos(incl_angle)]
    ], dtype='f4')

    # Offset orbital
    offset = orbit_inclined_rot @ orbit_rot @ np.array([planet.orbit_radius, 0, 0], dtype='f4')

    # Centre réel
    offset[2] *= -1
    res = sun.center + offset
    
    return res 



def generate_circle_points(segments=64):
    angle_step = 2 * np.pi / segments
    points = []
    for i in range(segments):
        angle = i * angle_step
        x = np.cos(angle)
        z = np.sin(angle)
        points.append([x, 0.0, z])
    return np.array(points, dtype='f4')


def generate_sphere_points(num_points, radius=1):
    points = np.empty((0,3))
    offset = 2 / num_points
    increment = math.pi * (3 - math.sqrt(5))+10.01  # angle d or
    for i in range(num_points):
        y = i * offset - 1 + (offset / 2)
        r = math.sqrt(1 - y*y)
        phi = i * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points = np.append(points, [[x*radius, y*radius, z*radius]], axis=0)

    return points


def generate_indexed_uv_sphere(radius=1.0, rings=16, segments=32, center=(0,0,0)):
    vertices = []
    indices = []

    for i in range(rings + 1):
        theta = np.pi * i / rings
        for j in range(segments + 1):
            phi = 2 * np.pi * j / segments
            x = radius * np.sin(theta) * np.cos(phi) + center[0]
            y = radius * np.cos(theta) + center[1]
            z = radius * np.sin(theta) * np.sin(phi) + center[2]
            vertices.append([x, y, z])

    for i in range(rings):
        for j in range(segments):
            first = i * (segments + 1) + j
            second = first + segments + 1

            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4')

class Camera():

    def __init__(self,pos=np.array([0.0, 0.0, -120.0], dtype='f4'),move_speed=5.0,rot_speed = 60.0,aspect_ratio=800/600):
        self.camera_pos = pos
        self.forward = np.array([0.0, 0.2, 1.0], dtype='f4')  
        self.forward_on_orbit = self.forward
        self.is_on_orbit = False
        self.up = np.array([0.0, 1.0, 0.0], dtype='f4')
        self.move_speed = move_speed 
        self.rot_speed = rot_speed
        self.yaw = 0.0   
        self.pitch = 0.0
        self.aspect_ratio = aspect_ratio
        self.t = 1
        self.change = True
        self.offset = -1
        self.last_for_orbit = self.forward_on_orbit
        self.orbit_planet_last_pos = [0,0,0]

    def get_forward_vector(self):
        
        #v1 with pitch 
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        
        
        x = np.cos(pitch_rad) * np.sin(yaw_rad)
        y = np.sin(pitch_rad)
        z = np.cos(pitch_rad) * np.cos(yaw_rad)
        vec = np.array([x, y, z], dtype='f4')
        self.forward =  vec / np.linalg.norm(vec)  
        #print(self.yaw,self.pitch,self.forward)


    def HandleRotation(self,key_press,dt):
        if key_press.get(pygame.K_RIGHT):
            self.yaw -= self.rot_speed * dt  
            
        if key_press.get(pygame.K_LEFT):
            self.yaw += self.rot_speed * dt
        if key_press.get(pygame.K_UP):
            self.pitch += self.rot_speed * dt
        if key_press.get(pygame.K_DOWN):
            self.pitch -= self.rot_speed * dt
        self.pitch = np.clip(self.pitch, -89.0, 89.0)


      
        
        self.get_forward_vector()


    def HandleMovement(self,key_press,dt):
        self.right = np.cross(self.forward, self.up)
        if np.linalg.norm(self.right)!= 0:
        
            self.right /= np.linalg.norm(self.right)


        # vecteur forward actuel
        if self.is_on_orbit:
            self.right *= self.offset
            v = self.right.copy()

            # axe de rotation

            axis =  self.up/np.linalg.norm(self.up)

            # angle en radians
            angle = np.radians(90.0)

            # Rodrigues formula inline
            c = np.cos(angle)
            s = np.sin(angle)
            self.forward = v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
            if np.linalg.norm(self.forward)!= 0:
                
           
                self.forward /= np.linalg.norm(self.forward)

            if (self.last_for_orbit[1] < 0 and self.forward[1] > 0) or (self.last_for_orbit[1] > 0 and self.forward[1] < 0 ): 
                #print("change",self.forward)
    
                self.offset = -1
                self.right *= self.offset
                self.forward *= self.offset
            self.last_for_orbit = self.forward
            
        if key_press.get(pygame.K_z):
            self.camera_pos += self.forward * self.move_speed * dt


        if key_press.get(pygame.K_s) :
            self.camera_pos -= self.forward * self.move_speed * dt



        if key_press.get(pygame.K_d):
            self.camera_pos += self.right * self.move_speed * dt
        if key_press.get(pygame.K_q):
            self.camera_pos -= self.right * self.move_speed * dt
        if key_press.get(pygame.K_SPACE):
            self.camera_pos += self.up * self.move_speed * dt
        if key_press.get(pygame.K_LSHIFT):
            self.camera_pos -= self.up * self.move_speed * dt
        if key_press.get(pygame.K_1):
            self.move_speed = 10/1000
        if key_press.get(pygame.K_2):
            self.move_speed = 100/1000
        if key_press.get(pygame.K_3):
            self.move_speed = 500/1000
        if key_press.get(pygame.K_4):
            self.move_speed = 1000/1000
        if key_press.get(pygame.K_5):
            self.move_speed = 1500/1000
        if key_press.get(pygame.K_6):
            self.move_speed = 2500/10
        if key_press.get(pygame.K_7):
            self.move_speed = 10000/10
        if key_press.get(pygame.K_8):
            self.move_speed = 70000/10
        if key_press.get(pygame.K_e):
            self.t = 15
        if key_press.get(pygame.K_a):
            self.t = 1


    def UpdateForProj(self):
        self.projection = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.01, 3000.0)
        
        self.projection_circle = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.01, 1500.0)

        if self.is_on_orbit == False:
            self.target = self.camera_pos + self.forward
        else:
            
            self.target = self.camera_pos - self.up
        self.view = Matrix44.look_at(self.camera_pos, self.target, np.array([0.0, 1.0, 0.0], dtype='f4'))