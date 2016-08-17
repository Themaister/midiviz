#version 310 es
layout(location = 0) in vec4 Position;
layout(location = 0) out mediump vec2 vCoord;

void main()
{
   gl_Position = Position;
   vCoord = vec2(1.0, -1.0) * Position.xy * 0.5 + 0.5;
}
