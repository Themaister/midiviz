#ifndef COMPUTE_INC_H
#define COMPUTE_INC_H

layout(local_size_x = 64) in;

layout(push_constant, std430) uniform Constants
{
   uint base;
   uint mask;
#ifdef BASE_STRUCT
   BASE_STRUCT params;
#endif
} registers;

layout(std430, set = 0, binding = 0) buffer Positions
{
   vec2 elems[];
} positions;

layout(std430, set = 0, binding = 1) buffer Velocities 
{
   uint elems[];
} velocities;

layout(std430, set = 0, binding = 2) buffer Colors 
{
   uvec2 elems[];
} colors;

vec2 unpack(uint v)
{
   return unpackHalf2x16(v);
}

vec4 unpack(uvec2 v)
{
   return vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y));
}

uint pack(vec2 v)
{
   return packHalf2x16(v);
}

uvec2 pack(vec4 v)
{
   return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}

void write_position(vec2 pos)
{
   positions.elems[(registers.base + gl_GlobalInvocationID.x) & registers.mask] = pos;
}

void write_velocity(vec2 v)
{
   velocities.elems[(registers.base + gl_GlobalInvocationID.x) & registers.mask] = pack(v);
}

void write_color(vec4 v)
{
   colors.elems[(registers.base + gl_GlobalInvocationID.x) & registers.mask] = pack(v);
}

vec2 read_position()
{
   return positions.elems[(registers.base + gl_GlobalInvocationID.x) & registers.mask];
}

vec2 read_velocity()
{
   return unpack(velocities.elems[(registers.base + gl_GlobalInvocationID.x) & registers.mask]);
}

vec4 read_color()
{
   return unpack(colors.elems[(registers.base + gl_GlobalInvocationID.x) & registers.mask]);
}
#endif

