#version 310 es
layout(location = 0) in vec2 Position;
layout(location = 1) in vec4 Color;
layout(location = 0) out mediump vec3 vColor;

layout(std430, push_constant) uniform Constants
{
   vec2 scale;
   float point_scale;
} registers;

void main()
{
   gl_Position = Color.a < 0.025 ? vec4(0.0, 0.0, 0.0, -1.0) : vec4(registers.scale * Position, 0.0, 1.0);
   gl_PointSize = registers.point_scale * Color.a;
   vColor = Color.rgb * clamp(Color.a, 0.0, 1.0);
}
