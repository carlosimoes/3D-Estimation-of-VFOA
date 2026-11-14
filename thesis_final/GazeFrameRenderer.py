"""Gaze arrow rendering for the feature videos (legacy module).

Two renderers are available:

* :func:`draw_gaze_arrow` - a pure OpenCV 2D arrow (the default; needs no GPU,
  no OpenGL and no display). It returns a NEW image and never draws in place.
* :class:`OpenGLVectorRenderer` - the thesis ray-marched 3D arrow, rendered
  with PyOpenGL in a ``lucid`` offscreen context. ``lucid``/``OpenGL`` are
  imported lazily in :func:`_gl`; when they (or a GL context) are unavailable
  ``compileShader()`` returns False and ``renderArrow`` falls back to
  :func:`draw_gaze_arrow` on a black canvas.

Directions are camera-frame vectors (OpenCV pinhole: +X right, +Y down, +Z
forward); positions and lengths are pixels unless stated otherwise.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

import logging
import math
import types
from typing import Optional, Sequence

import cv2
import numpy as np
from numpy import float32, frombuffer, uint8

logger = logging.getLogger(__name__)

VERTEX_POSITIONS = float32([[-1, -1], [1, -1], [-1, 1], [1, 1]])
GL_EXTENSION = "#extension GL_ARB_explicit_attrib_location: enable"

_GL_NAMES = (
    "GL_VERTEX_SHADER", "GL_COLOR_BUFFER_BIT", "glEnableVertexAttribArray", "GL_FLOAT", "glUniform3f",
    "GL_FRAGMENT_SHADER", "glUniform1f", "glDrawArrays", "glGetUniformLocation", "GL_RGB", "glReadPixels",
    "glClear", "glVertexAttribPointer", "GL_TRIANGLE_STRIP", "GL_UNSIGNED_BYTE",
)


def _gl() -> types.SimpleNamespace:
    """Import lucid/PyOpenGL lazily and return a namespace of the GL names used here.

    Raises ImportError when ``lucid`` or ``PyOpenGL`` is not installed.
    """
    from lucid.misc.gl.glcontext import create_opengl_context  # noqa: PLC0415
    import OpenGL.GL as GL  # noqa: PLC0415
    from OpenGL.GL.shaders import compileProgram, compileShader  # noqa: PLC0415

    ns = types.SimpleNamespace(
        create_opengl_context=create_opengl_context,
        compileProgram=compileProgram,
        compileShader=compileShader,
    )
    for name in _GL_NAMES:
        setattr(ns, name, getattr(GL, name))
    return ns


def draw_gaze_arrow(
    image: np.ndarray,
    origin_px: Sequence[float],
    direction: Sequence[float],
    length_px: float = 80.0,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Return a COPY of ``image`` with a 2D gaze arrow drawn on it.

    ``origin_px`` is (u, v) in pixels; ``direction`` is a camera-frame 3D vector
    (+X right, +Y down), normalised in 3D, so the arrow goes from the origin to
    ``origin + length_px * d[:2]`` (a direction along the optical axis gives only
    the origin dot). A filled radius-3 dot marks the origin. Non-finite or zero
    inputs return a plain copy.
    """
    out = np.array(image, dtype=np.uint8, copy=True, order="C")
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    try:
        o = np.asarray(origin_px, dtype=np.float64).reshape(-1)[:2]
        d = np.asarray(direction, dtype=np.float64).reshape(-1)[:3]
        length = float(length_px)
    except (TypeError, ValueError):
        return out
    if o.size < 2 or d.size < 3:
        return out
    if not (np.isfinite(o).all() and np.isfinite(d).all() and math.isfinite(length)):
        return out
    n = float(np.linalg.norm(d))
    if n <= 1e-12:
        return out
    d = d / n
    lim = 10.0 * max(out.shape[0], out.shape[1], 1)
    o = np.clip(o, -lim, lim)
    end = np.clip(o + length * d[:2], -lim, lim)
    p0 = (int(round(o[0])), int(round(o[1])))
    p1 = (int(round(end[0])), int(round(end[1])))
    col = tuple(int(c) for c in color)
    if p1 != p0:
        cv2.arrowedLine(out, p0, p1, col, int(thickness), cv2.LINE_AA, 0, 0.25)
    cv2.circle(out, p0, 3, col, -1, cv2.LINE_AA)
    return out


"""
Renders a 3D vector using PyOpenGL interface, executes parsed C++ code.
The method renderArrow returns an array of shape (outputWidth, outputHeight,3) containing a 3D arrow, given 3D position and vector components.
Before one or several calls of renderArrow, the method compileShader must be called.
"""


class OpenGLVectorRenderer:
    def __init__(self, outputWidth: int, outputHeight: int, glVersion: int = 140):
        """
        :param glVersion: must follow the pattern "#version {n}", with n an integer equal to your OpenGL shading language version (type glxinfo in terminal to get it) :
                        if 1.00, the argument mlust be "100;if 1.40, "140".
        """

        self.width, self.height = int(outputWidth), int(outputHeight)
        self.glVersion = "#version "+str(glVersion)
        self.shader = None
        self.xpos = None
        self.ypos = None
        self.vdir_x = None
        self.vdir_y = None
        self.vdir_z = None
        self.arrow_size = None
        self.res_loc = None
        self._gl: Optional[types.SimpleNamespace] = None

    def compileShader(self) -> bool:
        """
        Creates an OpenGL context, compiles the shaders and sets the uniform locations.
        Returns True on success, or False (with a warning) when lucid/OpenGL or a GL
        context is unavailable; renderArrow then uses the cv2 fallback. Never raises.
        """
        try:
            gl = _gl()
            gl.create_opengl_context((self.width, self.height))
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            self._gl = gl
            self.shader = gl.compileProgram(self.compileVertexShader(), self.compileFragmentShader())
            self.xpos = gl.glGetUniformLocation(self.shader, 'xpos')
            self.ypos = gl.glGetUniformLocation(self.shader, 'ypos')
            self.vdir_x = gl.glGetUniformLocation(self.shader, 'vdir_x')
            self.vdir_y = gl.glGetUniformLocation(self.shader, 'vdir_y')
            self.vdir_z = gl.glGetUniformLocation(self.shader, 'vdir_z')
            self.arrow_size = gl.glGetUniformLocation(self.shader, 'size')
            self.res_loc = gl.glGetUniformLocation(self.shader, 'iResolution')
            return True
        except Exception as exc:  # ImportError, GL context or shader errors
            logger.warning("OpenGL arrow renderer unavailable (%s); using the cv2 arrow fallback", exc)
            self._gl = None
            self.shader = None
            return False

    def renderArrow(self, position, vector, asize):
        """
        Returns a (height, width, 3) uint8 image with an arrow starting at ``position``
        (normalised image coordinates in [0, 1]^2) along the camera-frame ``vector``
        (+X right, +Y down, +Z forward). ``asize`` is the arrow size (the cv2 fallback
        draws an arrow of 2 * asize * width pixels).
        """
        gl = self._gl
        if gl is None or self.shader is None:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            try:
                origin = (float(position[0]) * self.width, float(position[1]) * self.height)
                length = 2.0 * float(asize) * self.width
            except (TypeError, ValueError, IndexError):
                return canvas
            return draw_gaze_arrow(canvas, origin, vector, length)
        # the shader works in the Gaze360 eye frame: x right, y up, -z towards the camera
        g = (float(vector[0]), -float(vector[1]), float(vector[2]))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        with self.shader:
            gl.glUniform1f(self.xpos, (2 * position[0] - 1) * 0.89)
            gl.glUniform1f(self.ypos, (-2 * position[1] + 1) * 0.67)
            gl.glUniform1f(self.vdir_x, -g[0])
            gl.glUniform1f(self.vdir_y, g[1])
            gl.glUniform1f(self.vdir_z, -g[2])
            gl.glUniform1f(self.arrow_size, asize)
            gl.glUniform3f(self.res_loc, self.width, self.height, 1.0)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, VERTEX_POSITIONS)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        return frombuffer(pixels, uint8).reshape(self.height, self.width, 3)[::-1].copy()

    def compileVertexShader(self):
        gl = self._gl if self._gl is not None else _gl()
        return gl.compileShader(self.glVersion + "\n" + GL_EXTENSION + "\n" + """
                layout(location = 0) in vec4 position;
                out vec2 UV;
                void main()
                {
                  UV = position.xy*0.5+0.5;
                  gl_Position = position;
                }
                """
                             , gl.GL_VERTEX_SHADER)

    def compileFragmentShader(self):
        gl = self._gl if self._gl is not None else _gl()
        return gl.compileShader(self.glVersion + "\n" + GL_EXTENSION + "\n" + """
                out vec4 outputColor;
                in vec2 UV;
                
                uniform sampler2D iChannel0;
                uniform vec3 iResolution;
                vec4 iMouse = vec4(0);
                uniform float iTime = 0.0;
                uniform float xpos = 100.0;
                uniform float ypos = 0.1;
                uniform float size = 10.0;
                uniform float vdir_x = -1.0;
                uniform float vdir_y = -1.0;
                uniform float vdir_z = 1.0;
                const vec3 X = vec3(1., 0., 0.);
                const vec3 Y = vec3(0., 1., 0.);
                const vec3 Z = vec3(0., 0., 1.);
                
                // YOURS
                const float Z_NEAR = 1.0;
                const float Z_FAR  = 400.0;
                const float EPSILON = 0.01;
                const float FOCAL_LENGTH = 30.0;
                const vec3 EYE_LOOK_POINT = vec3(0, 0, 5);
                const vec3 WHITE = vec3(1, 1, 1);
                const vec3 BLACK = vec3(0, 0, 0);
                const vec3 RED =   vec3(1, 0, 0);
                const vec3 GREEN = vec3(0, 1, 0);
                const vec3 BLUE =  vec3(0, 0, 1);
                const vec3 TOP_BG_COLOR = WHITE;
                const vec3 BOT_BG_COLOR = GREEN;
                const vec3 AMBIANT_COLOR = WHITE;
                const vec3 SPECULAR_COLOR = WHITE;
                const float AMBIANT_RATIO = 0.3;
                const float DIFFUSE_RATIO = 0.8;
                const float SPECULAR_RATIO = 0.4;
                const float SPECULAR_ALPHA = 5.;
                const vec3 LIGHT_DIRECTION = normalize(vec3(1, -1, -1));
                
                vec2 normalizeAndCenter(in vec2 coord) {
                    return (2.0 * coord - iResolution.xy) / iResolution.y;
                }
                
                vec3 rayDirection(vec3 eye, vec2 uv) {    
                    vec3 z = normalize(eye - EYE_LOOK_POINT);
                    vec3 x = normalize(cross(Y, z));
                    vec3 y = cross(z, x);
                
                    return normalize(
                          x * uv.x 
                        + y * uv.y 
                        - z * FOCAL_LENGTH);
                }
                
                //
                // Rotations
                //
                
                vec3 rotX(vec3 point, float angle) {
                    mat3 matRotX = mat3(
                        1.0, 0.0, 0.0, 
                        0.0, cos(angle), -sin(angle), 
                        0.0, sin(angle), cos(angle));
                    return matRotX * point;
                }
                
                vec3 rotY(vec3 point, float angle) {
                    mat3 matRotY = mat3( 
                        cos(angle*0.5), 0.0, -sin(angle*0.5),
                        0.0, 1.0, 0.0, 
                        sin(angle*0.5), 0.0, cos(angle*0.5));
                    return matRotY * point;
                }
                
                vec3 rotZ(vec3 point, float angle) {
                    mat3 matRotZ = mat3(
                        cos(angle*0.1), -sin(angle*0.1), 0.0, 
                        sin(angle*0.1), cos(angle*0.1), 0.0,
                        0.0, 0.0, 1.0);
                    return matRotZ * point;
                }
                
                //
                // Positioning
                //
                
                vec3 randomOrtho(vec3 v) {
                    if (v.x != 0. || v.y != 0.) {
                        return normalize(vec3(v.y, -v.x, 0.));
                    } else {
                        return normalize(vec3(0., v.z, -v.y));
                    }
                } 
                
                vec3 atPosition(vec3 point, vec3 position) {
                    return (point - position);
                }
                
                vec3 atCoordSystem(vec3 point, vec3 center, vec3 dx, vec3 dy, vec3 dz) {
                    vec3 localPoint = (point - center);
                    return vec3(
                        dot(localPoint, dx),
                        dot(localPoint, dy),
                        dot(localPoint, dz));
                }
                
                vec3 atCoordSystemX(vec3 point, vec3 center, vec3 dx) {
                    vec3 dy = randomOrtho(dx);
                    vec3 dz = cross(dx, dy);
                    return atCoordSystem(point, center, dx, dy, dz);
                }
                
                vec3 atCoordSystemY(vec3 point, vec3 center, vec3 dy) {
                    vec3 dz = randomOrtho(dy);
                    vec3 dx = cross(dy, dz);
                    return atCoordSystem(point, center, dx, dy, dz);
                }
                
                vec3 atCoordSystemZ(vec3 point, vec3 center, vec3 dz) {
                    vec3 dx = randomOrtho(dz);
                    vec3 dy = cross(dz, dx);
                    return atCoordSystem(point, center, dx, dy, dz);
                }
                
                //
                // Shapes
                //
                
                float capsule(vec3 coord, float height, float radius)
                {
                    coord.y -= clamp( coord.y, 0.0, height );
                    return length( coord ) - radius;
                }
                
                float roundCone(vec3 coord, in float radiusTop, float radiusBot, float height)
                {
                    vec2 q = vec2( length(coord.xz), coord.y );
                    float b = (radiusBot-radiusTop)/height;
                    float a = sqrt(1.0-b*b);
                    float k = dot(q,vec2(-b,a));
                    if( k < 0.0 ) return length(q) - radiusBot;
                    if( k > a*height ) return length(q-vec2(0.0,height)) - radiusTop;
                    return dot(q, vec2(a,b) ) - radiusBot;
                }
                
                //
                // Boolean ops
                //
                
                vec4 shape(float dist, vec3 color) {
                    return vec4(color, dist);
                }
                
                vec4 join(vec4 shape1, vec4 shape2) {
                    if (shape1.a < shape2.a) {
                        return shape1;
                    } else {
                        return shape2;
                    }
                }
                
                vec4 join(vec4 shape1, vec4 shape2, vec4 shape3) {
                    return join(join(shape1, shape2), shape3);
                }
                
                vec4 join(vec4 shape1, vec4 shape2, vec4 shape3, vec4 shape4) {
                    return join(join(shape1, shape2, shape3), shape4);
                }
                
                //
                // Scene
                // x range: 355
                // y rangeL 205
                
                vec4 dist(in vec3 coord) {
                    vec3 V_y = normalize(vec3(vdir_x,vdir_y,vdir_z));
                    vec3 V_x = randomOrtho(V_y);
                    vec3 V_z = cross(V_x,V_y);
                    vec3 pos = vec3(xpos,ypos,0);
                    vec3 offset = pos+V_y*size*4.0;
                    vec3 ARM_2_X = V_x;
                    vec3 ARM_2_Y = V_y;
                    vec3 ARM_2_Z = V_z;
                    vec3 NOZE_O = vec3(0,11,0);
                    vec3 NOZE_X = -V_x;
                    vec3 NOZE_Y = V_y;
                    vec3 NOZE_Z = V_z;
                    vec4 noze = shape(roundCone(atCoordSystem(coord, offset, NOZE_X, NOZE_Y, NOZE_Z), size*0.05, size*0.4, size*2.0), RED);
                    vec4 rightArm = shape(capsule(atCoordSystem(coord, pos, ARM_2_X, ARM_2_Y, ARM_2_Z), size*5., size*0.1), RED);
                    return join(noze, rightArm);
                }
                
                //
                //
                //
                
                bool rayMarching(in vec3 startPoint, in vec3 direction, out vec3 lastPoint, out vec3 color) {
                    lastPoint = startPoint;
                    for (int i = 0; i < 50; ++i) {
                        vec4 d = dist(lastPoint);
                        if (d.a < EPSILON) {
                            color = d.xyz;
                            return true;
                        } else {
                            lastPoint += d.a * direction;
                        }
                        if (lastPoint.z < -Z_FAR) {
                            break;
                        }
                    }    
                    return false;
                }
                
                vec3 norm(in vec3 coord) {
                    vec3 eps = vec3( EPSILON, 0.0, 0.0 );
                    vec3 nor = vec3(
                        dist(coord+eps.xyy).a - dist(coord-eps.xyy).a,
                        dist(coord+eps.yxy).a - dist(coord-eps.yxy).a,
                        dist(coord+eps.yyx).a - dist(coord-eps.yyx).a);
                        return normalize(nor);
                }
                
                vec3 cellShadingObjColor(vec3 point, vec3 ray, vec3 objColor) {
                    vec3 n = norm(point);
                    float diffuseValue = max(dot(-LIGHT_DIRECTION, n), 0.);
                    float specularValue = pow(max(dot(-reflect(LIGHT_DIRECTION, n), ray), 0.), SPECULAR_ALPHA);        
                    return AMBIANT_COLOR * AMBIANT_RATIO
                        + objColor * DIFFUSE_RATIO * diffuseValue
                        + SPECULAR_COLOR * SPECULAR_RATIO * specularValue;
                }
                
                vec3 computeColor(vec2 fragCoord) {
                    vec2 uv = normalizeAndCenter(fragCoord);
                    vec3 eye = vec3(0, 0, 20);
                    vec3 ray = rayDirection(eye, uv);
                    vec3 intersection;
                    vec3 color;
                    bool intersected = rayMarching(eye, ray, intersection, color);
                    if (intersected) {
                        return cellShadingObjColor(intersection, ray, color);
                    } else {
                        return vec3(0,0,0);
                    }
                }
                
                //#define SUPERSAMPLING
                void mainImage( out vec4 fragColor, in vec2 fragCoord )
                {    
                    #ifdef SUPERSAMPLING
                    fragColor = vec4(0);
                    float count = 0.;
                    for(float dx=-0.5; dx<=0.5; dx+=0.5) {
                        for(float dy=-0.5; dy<=0.5; dy+=0.5) {
                            fragColor += vec4(computeColor(fragCoord + vec2(dx, dy)), 1.0);
                            count += 1.;
                        }
                    }
                    fragColor /= count;
                    #else
                    fragColor = vec4(computeColor(fragCoord),1.0);
                    #endif
                } 
                
                void main()
                {
                    mainImage(outputColor, UV*iResolution.xy);
                }
                
                """,
                             gl.GL_FRAGMENT_SHADER)
