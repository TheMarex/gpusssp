#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main()
{
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = length(coord);
    
    float alpha = 1.0 - smoothstep(0.85, 1.0, dist);
    
    if (alpha < 0.01)
    {
        discard;
    }
    
    outColor = vec4(fragColor, alpha);
}
