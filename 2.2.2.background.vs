#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 envRotationMatrix;

void main()
{
    TexCoords = aPos;
    
    // Remove translation from the view matrix so background doesn't move when camera moves
    mat4 staticView = mat4(mat3(view)); 
    
    // Rotate the skybox position based on input matrix
    vec4 rotatedPos = envRotationMatrix * vec4(aPos, 1.0);
    
    vec4 pos = projection * staticView * vec4(rotatedPos.xyz, 1.0);
    gl_Position = pos.xyww; // Force depth optimization trick
}