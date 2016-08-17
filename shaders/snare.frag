#version 310 es
precision mediump float;

layout(location = 0) in mediump vec2 vCoord;
layout(location = 0) out vec4 FragColor;

layout(std430, push_constant) uniform Constants
{
   float vel;
} registers;

void main()
{
   float gray = smoothstep(0.7, 1.0, vCoord.y);
   float alpha = 0.5 * registers.vel * vCoord.y;
   FragColor = vec4(vec3(gray), alpha);
}
