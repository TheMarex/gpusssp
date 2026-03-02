#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform PushConstants
{
    vec2 offset;
    vec2 scale;
    float point_size;
}
pc;

void main()
{
    gl_Position = vec4((inPosition + pc.offset) * pc.scale, 0.0, 1.0);
    gl_PointSize = pc.point_size;
    fragColor = inColor;
}
