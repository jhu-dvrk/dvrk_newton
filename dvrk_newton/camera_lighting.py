"""Lights mounted above and below the ECM optical center."""

from __future__ import annotations

import numpy as np

from dvrk_simulator_base.types import Pose


LIGHT_OFFSET_M = 0.004


def endoscope_light_positions(optical_pose: Pose) -> np.ndarray:
    """Return the upper and lower lights in world coordinates (optical +Y is down)."""
    up = -optical_pose.orientation[:, 1]
    return np.asarray(
        [optical_pose.position + LIGHT_OFFSET_M * up,
         optical_pose.position - LIGHT_OFFSET_M * up],
        dtype=np.float32,
    )


# Keep Newton's material, environment, tone mapping, and transparency handling;
# replace its single sun with two local lights in the camera renderer only.
_DIRECT_LIGHTING = """
    vec3 N = normalize(Normal);
    vec3 V = normalize(-camera_to_fragment);
    if (!gl_FrontFacing) N = -N;
    float NdotV = max(dot(N, V), 0.001);
    vec3 F_max = mix(F0, vec3(1.0), 1.0 - roughness);
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 2; ++i) {
        vec3 to_light = endoscope_positions[i] - camera_to_fragment;
        float distance_squared = dot(to_light, to_light);
        vec3 L = normalize(to_light);
        vec3 H = normalize(V + L);
        float cone = clamp((dot(-L, endoscope_forward) - 0.85) / 0.10, 0.0, 1.0);
        float attenuation = cone / (1.0 + 0.02 * distance_squared);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);
        float a = roughness * roughness;
        float a2 = a * a;
        float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = a2 / (PI * denom * denom);
        float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        float G = (NdotV / (NdotV * (1.0 - k) + k))
                * (NdotL / (NdotL * (1.0 - k) + k));
        vec3 F = F0 + (F_max - F0) * pow(1.0 - HdotV, 5.0);
        vec3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);
        vec3 diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
        Lo += (diffuse * diffuse_scale + spec * specular_scale)
              * NdotL * attenuation * 1.5;
    }

"""


def install_opengl_endoscope_lights(renderer) -> None:
    """Specialize each camera shader, including lazily created transparent shaders.

    Newton 1.6 has no ViewerGL API for local lights. Limit shader adaptation to
    this renderer instance; the interactive viewer and Newton sources are untouched.
    """
    from newton._src.viewer.gl import shaders
    from pyglet.graphics.shader import Shader, ShaderProgram

    source = shaders.shape_fragment_shader
    start = source.index("    // surface vectors")
    end = source.index("    // Hemispherical ambient", start)
    source = source[:start] + _DIRECT_LIGHTING + source[end:]
    source = source.replace(
        "uniform vec3 light_color;",
        "uniform vec3 light_color;\nuniform vec3 endoscope_positions[2];\n"
        "uniform vec3 endoscope_forward;",
    )
    original_update = renderer._update_shape_shader
    adapted = set()

    def update_shape_shader(shader):
        if shader not in adapted:
            vertex, fragment = shaders.shape_vertex_shader, source
            if shader.loc_oit_enabled is not None:
                vertex = shaders._with_shader_define(vertex, "ENABLE_TRANSPARENCY")
                fragment = shaders._with_shader_define(fragment, "ENABLE_TRANSPARENCY")
            original = shader.shader_program
            names = {data["location"]: name for name, data in original.uniforms.items()}
            shader.shader_program = ShaderProgram(
                Shader(vertex, "vertex"), Shader(fragment, "fragment")
            )
            # The new program assigns different locations to existing uniforms.
            for attribute, location in vars(shader).copy().items():
                if attribute.startswith("loc_") and location is not None and location >= 0:
                    setattr(shader, attribute, shader._get_uniform_location(names[location]))
            original.delete()
            adapted.add(shader)
        original_update(shader)
        positions, forward = renderer.endoscope_lights
        with shader:
            for i, position in enumerate(positions):
                shader._gl.glUniform3f(
                    shader._get_uniform_location(f"endoscope_positions[{i}]"), *position
                )
            shader._gl.glUniform3f(
                shader._get_uniform_location("endoscope_forward"), *forward
            )

    renderer._update_shape_shader = update_shape_shader
