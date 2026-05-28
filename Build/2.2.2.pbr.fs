#version 330 core
out vec4 FragColor;
in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

uniform mat4 envRotationMatrix;

// material parameters
uniform sampler2D texture_diffuse1;
uniform sampler2D texture_normal1;
uniform sampler2D texture_roughness1; // Packed ARM texture (R=AO, G=Roughness, B=Metallic)

// IBL
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

// Dynamic Directional Light (Updated via C++ right-click drag)
uniform vec3 lightDirection;

uniform vec3 camPos;

const float PI = 3.14159265359;

// ----------------------------------------------------------------------------
// Reconstructs world-space normals from the normal map using screen-space derivatives
vec3 getNormalFromMap()
{
    vec3 tangentNormal = texture(texture_normal1, TexCoords).xyz * 2.0 - 1.0;

    vec3 Q1  = dFdx(WorldPos);
    vec3 Q2  = dFdy(WorldPos);
    vec2 st1 = dFdx(TexCoords);
    vec2 st2 = dFdy(TexCoords);

    vec3 N   = normalize(Normal);
    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ----------------------------------------------------------------------------
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}   

// ----------------------------------------------------------------------------
void main()
{        
    // 1. Unpack material properties (sRGB albedo conversion + ARM unpacking)
    vec3 albedo = pow(texture(texture_diffuse1, TexCoords).rgb, vec3(2.2));
    vec3 arm = texture(texture_roughness1, TexCoords).rgb;
    
    float ao        = arm.r; 
    float roughness = arm.g;
    float metallic  = arm.b;
       
    // 2. Setup geometric vectors
    vec3 N = getNormalFromMap();
    vec3 V = normalize(camPos - WorldPos);
    vec3 R = reflect(-V, N); 

    // Calculate surface reflectance at normal incidence (F0)
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // 3. Direct Lighting: Dynamic Directional Light
    vec3 Lo = vec3(0.0);
    {
        // Vector L points towards the light source
        vec3 L = normalize(-lightDirection); 
        vec3 H = normalize(V + L);
        
        // Directional light radiance (No attenuation over distance)
        vec3 radiance = vec3(3.0); // Adjust this intensity value to change sun brightness

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);    
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);        
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // prevent divide by zero
        vec3 specular = numerator / denominator;
        
        // Energy conservation
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;                    
            
        float NdotL = max(dot(N, L), 0.0);        

        // Add to outgoing direct radiance
        Lo += (kD * albedo / PI + specular) * radiance * NdotL; 
    }   
    
    // 4. Indirect Lighting: Image-Based Ambient Lighting (IBL)
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;      
    
    // Rotate the lookup directions based on left-click interaction
    vec3 rotatedN = vec3(envRotationMatrix * vec4(N, 0.0));
    vec3 rotatedR = vec3(envRotationMatrix * vec4(R, 0.0));

    // Sample Diffuse IBL
    vec3 irradiance = texture(irradianceMap, rotatedN).rgb;
    vec3 diffuse    = irradiance * albedo;
    
    // Sample Specular IBL (Split-Sum Approximation)
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, rotatedR, roughness * MAX_REFLECTION_LOD).rgb;    
    vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    // Combine IBL ambient components
    vec3 ambient = (kD * diffuse + specular) * ao;
    
    // 5. Final Composite (Direct + Ambient)
    vec3 color = ambient + Lo;

    // HDR Reinhard tonemapping
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0/2.2)); 

    FragColor = vec4(color, 1.0);
}