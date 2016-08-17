#version 310 es
precision mediump float;

layout(location = 0) in mediump vec3 vColor;
layout(location = 0) out vec4 FragColor;

void main()
{
   vec2 mid = gl_PointCoord - 0.5;
   float dst_sqr = dot(mid, mid);
   float a = exp2(-20.0 * dst_sqr);
   FragColor = vec4(vColor, a);
}
