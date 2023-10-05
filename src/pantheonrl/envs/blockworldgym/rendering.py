"""
2D rendering framework
"""
import os
import sys

import math
import numpy as np

from gymnasium import error


if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    """
    ) from e

try:
    from pyglet import gl
except ImportError as e:
    raise ImportError(
        """
    Error occurred while running `from pyglet.gl import *`
    """
    ) from e

RAD2DEG = 57.29577951308232


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    if isinstance(spec, str):
        return pyglet.canvas.Display(spec)

    raise error.Error(
        f"Invalid display specification: {spec}. \
        (Must be a string like :0 or None.)"
    )


def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs,
    )


class Viewer:
    """Viewer class from early gym versions"""

    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = get_window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        """Closes the viewer window."""
        if self.isopen and sys.meta_path:
            self.window.close()
            self.isopen = False

    def window_closed_by_user(self):
        """Called when window is closed by user."""
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        """Sets the bounds of the window"""
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley),
            scale=(scalex, scaley),
        )

    def add_geom(self, geom):
        """Adds the geometry to the scene"""
        self.geoms.append(geom)

    def add_onetime(self, geom):
        """Adds the geometry for one-time viewing"""
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        """Renders the scene"""
        gl.glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        """Draw circle"""
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        """Draw polygon"""
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        """Draw polyline"""
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        """Draw line"""
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        """Get the window as an rgb array"""
        self.window.flip()
        image_data = (
            pyglet.image.get_buffer_manager()
            .get_color_buffer()
            .get_image_data()
        )
        self.window.flip()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]

    def __del__(self):
        self.close()


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom:
    """Base Geometry class"""

    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        """Render the geometry"""
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        """Render the geometry, implemented by subclass"""
        raise NotImplementedError

    def add_attr(self, attr):
        """Add an attribute"""
        self.attrs.append(attr)

    def set_color(self, r, g, b):
        """Set the color"""
        self._color.vec4 = (r, g, b, 1)


class Attr:
    """Base attribute class"""

    def enable(self):
        """Enable the attribute"""
        raise NotImplementedError

    def disable(self):
        """Disable the attribute"""


class Transform(Attr):
    """Class defining an opengl transform"""

    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        gl.glPushMatrix()
        gl.glTranslatef(
            self.translation[0], self.translation[1], 0
        )  # translate to GL loc ppint
        gl.glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        gl.glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        gl.glPopMatrix()

    def set_translation(self, newx, newy):
        """Set the translation"""
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        """Set the rotation"""
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        """Set the scale"""
        self.scale = (float(newx), float(newy))


class Color(Attr):
    """Color attribute"""

    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        gl.glColor4f(*self.vec4)


class LineStyle(Attr):
    """Line style attribute"""

    def __init__(self, style):
        self.style = style

    def enable(self):
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineStipple(1, self.style)

    def disable(self):
        gl.glDisable(gl.GL_LINE_STIPPLE)


class LineWidth(Attr):
    """Line width attribute"""

    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        gl.glLineWidth(self.stroke)


class Point(Geom):
    """Point geometry"""

    def __init__(self):
        Geom.__init__(self)

    def render1(self):
        gl.glBegin(gl.GL_POINTS)  # draw point
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glEnd()


class FilledPolygon(Geom):
    """Filled Polygon geometry"""

    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        if len(self.v) == 4:
            gl.glBegin(gl.GL_QUADS)
        elif len(self.v) > 4:
            gl.glBegin(gl.GL_POLYGON)
        else:
            gl.glBegin(gl.GL_TRIANGLES)
        for p in self.v:
            gl.glVertex3f(p[0], p[1], 0)  # draw each vertex
        gl.glEnd()


def make_circle(radius=10, res=30, filled=True):
    """Convenience method for constructing a circle"""
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    return PolyLine(points, True)


def make_polygon(v, filled=True):
    """Convenience method for constructing a polygon"""
    if filled:
        return FilledPolygon(v)
    return PolyLine(v, True)


def make_polyline(v):
    """Convenience method for constructing a polyline"""
    return PolyLine(v, False)


def make_capsule(length, width):
    """Convenience method for constructing a capsule"""
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    """Compound Geometry object"""

    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    """Polyline Geometry object"""

    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINE_LOOP if self.close else gl.GL_LINE_STRIP)
        for p in self.v:
            gl.glVertex3f(p[0], p[1], 0)  # draw each vertex
        gl.glEnd()

    def set_linewidth(self, x):
        """Set the line width"""
        self.linewidth.stroke = x


class Line(Geom):
    """Line Geometry object"""

    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(*self.start)
        gl.glVertex2f(*self.end)
        gl.glEnd()


class Image(Geom):
    """Image Geometry object"""

    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(
            -self.width / 2,
            -self.height / 2,
            width=self.width,
            height=self.height,
        )


# ================================================================


class SimpleImageViewer:
    """Simple Image viewer class"""

    def __init__(self, display=None, maxwidth=500):
        self.window = None
        self.isopen = False
        self.display = get_display(display)
        self.maxwidth = maxwidth
        self.width = None
        self.height = None

    def imshow(self, arr):
        """Show the image given an rgb array"""
        if self.window is None:
            height, width, _channels = arr.shape
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = get_window(
                width=width,
                height=height,
                display=self.display,
                vsync=False,
                resizable=True,
            )
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert (
            len(arr.shape) == 3
        ), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            arr.shape[1],
            arr.shape[0],
            "RGB",
            arr.tobytes(),
            pitch=arr.shape[1] * -3,
        )
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST
        )
        texture = image.get_texture()
        texture.width = self.width
        texture.height = self.height
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0)  # draw
        self.window.flip()

    def close(self):
        """Close the viewer"""
        if self.isopen and sys.meta_path:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
