# Adapted from https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/renderer.py

"""Abstract class of a renderer and a factory function to create a renderer.

The renderer produces an RGB/depth image of a 3D mesh model in a specified pose
for given camera parameters and illumination settings.
"""


def create_renderer(width, height, renderer_type='cpp', mode='rgb+depth',
                    shading='phong', bg_color=(0.0, 0.0, 0.0, 0.0)):
    """A factory to create a renderer.

    Note: Parameters mode, shading and bg_color are currently supported only by
    the 'vispy' and 'python' renderers (renderer_type='vispy' or renderer_type='python').
    To render on a headless server, either 'vispy' or 'cpp' can be used.

    :param width: Width of the rendered image.
    :param height: Height of the rendered image.
    :param renderer_type: Type of renderer (options: 'vispy', 'cpp', 'python').
    :param mode: Rendering mode ('rgb+depth', 'rgb', 'depth').
    :param shading: Type of shading ('flat', 'phong').
    :param bg_color: Color of the background (R, G, B, A).
    :return: Instance of a renderer of the specified type.
    """
    if renderer_type == 'python':
        from bop_toolkit_lib import renderer_py
        return renderer_py.RendererPython(width, height, mode, shading, bg_color)
    elif renderer_type == "vispy":
        from cad2cosypose.custom_bop_toolkit_lib import renderer_vispy
        return renderer_vispy.RendererVispy(width, height, mode, shading, bg_color)
    elif renderer_type == 'cpp':
        from bop_toolkit_lib import renderer_cpp
        return renderer_cpp.RendererCpp(width, height)

    else:
        raise ValueError('Unknown renderer type.')
