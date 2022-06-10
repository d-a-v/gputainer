# https://pythonprogramming.net/opengl-rotating-cube-example-pyopengl-tutorial/

import time
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from pegl import egl


verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()



def create_initialized_headless_egl_display():
  """Creates an initialized EGL display directly on a device."""
  for device in EGL.eglQueryDevicesEXT():
    display = EGL.eglGetPlatformDisplayEXT(
        EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
    if display != EGL.EGL_NO_DISPLAY and EGL.eglGetError() == EGL.EGL_SUCCESS:
      # `eglInitialize` may or may not raise an exception on failure depending
      # on how PyOpenGL is configured. We therefore catch a `GLError` and also
      # manually check the output of `eglGetError()` here.
      try:
        initialized = EGL.eglInitialize(display, None, None)
      except error.GLError:
        pass
      else:
        if initialized == EGL.EGL_TRUE and EGL.eglGetError() == EGL.EGL_SUCCESS:
          return display
  return EGL.EGL_NO_DISPLAY 


def main():
    
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    print("GL_VENDOR =", glGetString(GL_VENDOR).decode())
    print("GL_RENDERER =", glGetString(GL_RENDERER).decode())
    print("GL_VERSION =", glGetString(GL_VERSION).decode())
    print("GL_SHADING_LANGUAGE_VERSION =", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())
    
    print("egl.egl_version = ", egl.egl_version)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)


    glTranslatef(0.0,0.0, -5)

    duration_sec = 5
    duration_ns = duration_sec * 1000000000
    start_ns = time.time_ns()
    counter = 0
    while time.time_ns() - start_ns < duration_ns:
        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        pygame.quit()
        #        quit()

        glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        #pygame.time.wait(10)
        counter = counter + 1
    print("img/s=", counter / duration_sec)
    
    


main()
