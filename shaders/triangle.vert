#version 310 es
layout(location = 0) in vec4 Position;
layout(location = 1) in vec4 Color;
layout(location = 0) out vec4 vColor;

layout(std430, push_constant) uniform UBO
{
   mat4 MVP;
} registers;

void main()
{
   gl_Position = registers.MVP * Position;
   vColor = Color;
}
