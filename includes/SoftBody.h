#ifndef SOFT_BODY_H
#define SOFT_BODY_H

#include <vector>
#include <unordered_map>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Forward declaration of your engine's Shader class
class Shader;

struct SoftBodyParticle {
    glm::vec3 pos;
    glm::vec3 prevPos;
    glm::vec3 vel;
    float invMass;
};

struct EdgeConstraint {
    int id0, id1;
    float restLength;
    float compliance;
    float lambda; // Tracks the accumulated Lagrange multiplier for XPBD accuracy
};

struct SoftBodyVertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

// Helper to hash glm::vec3 for quick vertex welding lookups
struct Vec3Hash {
    std::size_t operator()(const glm::vec3& v) const {
        // Simple hash combining with a small epsilon tolerance
        auto h1 = std::hash<float>()(std::round(v.x * 1000.0f));
        auto h2 = std::hash<float>()(std::round(v.y * 1000.0f));
        auto h3 = std::hash<float>()(std::round(v.z * 1000.0f));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class SoftBody {
public:
    std::vector<SoftBodyParticle> particles;
    std::vector<EdgeConstraint> constraints;
    std::vector<unsigned int> indices;

    unsigned int VAO, VBO, EBO;

    SoftBody() : VAO(0), VBO(0), EBO(0) {}

    // Destructor to clean up OpenGL memory resource leaks
    ~SoftBody() {
        cleanupBuffers();
    }

    // Generates a structural soft-body cube lattice with explicit compliance settings
    void InitializeCube(glm::vec3 center, glm::vec3 size, int resolution, float structuralCompliance = 0.0f, float shearCompliance = 0.005f) {
        particles.clear();
        constraints.clear();
        indices.clear();
        cleanupBuffers();

        int resX = resolution, resY = resolution, resZ = resolution;
        float dx = size.x / (resX - 1);
        float dy = size.y / (resY - 1);
        float dz = size.z / (resZ - 1);

        // 1. Create Particles
        for (int x = 0; x < resX; ++x) {
            for (int y = 0; y < resY; ++y) {
                for (int z = 0; z < resZ; ++z) {
                    SoftBodyParticle p;
                    p.pos = center + glm::vec3(x * dx, y * dy, z * dz) - size * 0.5f;
                    p.prevPos = p.pos;
                    p.vel = glm::vec3(0.0f);
                    p.invMass = 1.0f;
                    particles.push_back(p);
                }
            }
        }

        auto getIdx = [&](int x, int y, int z) {
            return x * (resY * resZ) + y * resZ + z;
            };

        // 2. Create Structural and Full Shear Springs
        for (int x = 0; x < resX; ++x) {
            for (int y = 0; y < resY; ++y) {
                for (int z = 0; z < resZ; ++z) {
                    int p0 = getIdx(x, y, z);

                    // Structural Neighbors
                    if (x < resX - 1) addConstraint(p0, getIdx(x + 1, y, z), structuralCompliance);
                    if (y < resY - 1) addConstraint(p0, getIdx(x, y + 1, z), structuralCompliance);
                    if (z < resZ - 1) addConstraint(p0, getIdx(x, y, z + 1), structuralCompliance);

                    // Complete Shear Diagonals 
                    if (x < resX - 1 && y < resY - 1) {
                        addConstraint(p0, getIdx(x + 1, y + 1, z), shearCompliance);
                        addConstraint(getIdx(x + 1, y, z), getIdx(x, y + 1, z), shearCompliance);
                    }
                    if (y < resY - 1 && z < resZ - 1) {
                        addConstraint(p0, getIdx(x, y + 1, z + 1), shearCompliance);
                        addConstraint(getIdx(x, y + 1, z), getIdx(x, y, z + 1), shearCompliance);
                    }
                    if (x < resX - 1 && z < resZ - 1) {
                        addConstraint(p0, getIdx(x + 1, y, z + 1), shearCompliance);
                        addConstraint(getIdx(x + 1, y, z), getIdx(x, y, z + 1), shearCompliance);
                    }
                }
            }
        }

        // 3. Build Triangles for Outer Surface Rendering
        for (int x = 0; x < resX - 1; ++x) {
            for (int y = 0; y < resY - 1; ++y) {
                for (int z = 0; z < resZ - 1; ++z) {
                    int bfl = getIdx(x, y, z);
                    int bfr = getIdx(x + 1, y, z);
                    int bbl = getIdx(x, y, z + 1);
                    int bbr = getIdx(x + 1, y, z + 1);
                    int tfl = getIdx(x, y + 1, z);
                    int tfr = getIdx(x + 1, y + 1, z);
                    int tbl = getIdx(x, y + 1, z + 1);
                    int tbr = getIdx(x + 1, y + 1, z + 1);

                    if (z == 0) {
                        indices.push_back(bfl); indices.push_back(tfl); indices.push_back(bfr);
                        indices.push_back(bfr); indices.push_back(tfl); indices.push_back(tfr);
                    }
                    if (z == resZ - 2) {
                        indices.push_back(bbl); indices.push_back(bbr); indices.push_back(tbl);
                        indices.push_back(bbr); indices.push_back(tbr); indices.push_back(tbl);
                    }
                    if (y == 0) {
                        indices.push_back(bfl); indices.push_back(bfr); indices.push_back(bbl);
                        indices.push_back(bfr); indices.push_back(bbr); indices.push_back(bbl);
                    }
                    if (y == resY - 2) {
                        indices.push_back(tfl); indices.push_back(tbl); indices.push_back(tfr);
                        indices.push_back(tfr); indices.push_back(tbl); indices.push_back(tbr);
                    }
                    if (x == 0) {
                        indices.push_back(bfl); indices.push_back(bbl); indices.push_back(tfl);
                        indices.push_back(tfl); indices.push_back(bbl); indices.push_back(tbl);
                    }
                    if (x == resX - 2) {
                        indices.push_back(bfr); indices.push_back(tfr); indices.push_back(bbr);
                        indices.push_back(tfr); indices.push_back(tbr); indices.push_back(bbr);
                    }
                }
            }
        }

        particleTexCoords.assign(particles.size(), glm::vec2(0.0f));
        setupBuffers();
    }

    void InitializeFromModel(const Model& model, const glm::vec3& spawnPos, const glm::vec3& scale, float compliance)
    {
        this->particles.clear();
        this->constraints.clear();
        this->indices.clear();
        this->particleTexCoords.clear(); // Clear old UV data
        this->cleanupBuffers();

        std::unordered_map<glm::vec3, unsigned int, Vec3Hash> vertexWeldMap;

        for (const Mesh& mesh : model.meshes)
        {
            std::vector<unsigned int> meshToGlobalIndexMap(mesh.vertices.size());

            for (size_t i = 0; i < mesh.vertices.size(); ++i)
            {
                glm::vec3 worldSpacePos = (mesh.vertices[i].Position * scale) + spawnPos;

                auto it = vertexWeldMap.find(worldSpacePos);
                if (it != vertexWeldMap.end())
                {
                    meshToGlobalIndexMap[i] = it->second;
                }
                else
                {
                    SoftBodyParticle p;
                    p.pos = worldSpacePos;
                    p.prevPos = p.pos;
                    p.vel = glm::vec3(0.0f);
                    p.invMass = 1.0f;

                    unsigned int newGlobalIndex = static_cast<unsigned int>(this->particles.size());
                    this->particles.push_back(p);

                    // Keep this line to save the original UV coordinates!
                    this->particleTexCoords.push_back(mesh.vertices[i].TexCoords);

                    vertexWeldMap[worldSpacePos] = newGlobalIndex;
                    meshToGlobalIndexMap[i] = newGlobalIndex;
                }
            }

            for (unsigned int localIndex : mesh.indices)
            {
                this->indices.push_back(meshToGlobalIndexMap[localIndex]);
            }
        }

        // 2. Generate Edge Constraints from Unique Mesh Edges
        std::unordered_set<unsigned long long> uniqueEdges;
        for (size_t i = 0; i < this->indices.size(); i += 3)
        {
            unsigned int i0 = this->indices[i];
            unsigned int i1 = this->indices[i + 1];
            unsigned int i2 = this->indices[i + 2];

            auto registerEdge = [&](unsigned int id0, unsigned int id1) {
                if (id0 == id1) return;
                unsigned int low = std::min(id0, id1);
                unsigned int high = std::max(id0, id1);
                unsigned long long edgeKey = (static_cast<unsigned long long>(low) << 32) | high;

                if (uniqueEdges.insert(edgeKey).second) {
                    this->addConstraint(low, high, compliance);
                }
                };

            registerEdge(i0, i1);
            registerEdge(i1, i2);
            registerEdge(i2, i0);
        }

        this->setupBuffers();

        std::cout << "[XPBD WELD LOG] Welded structural mesh down to " << particles.size()
            << " unique physical particles and " << uniqueEdges.size() << " edge constraints." << std::endl;
    }


    void Step(float dt, int subSteps, glm::vec3 gravity) {
        if (dt <= 0.0f || subSteps <= 0) return;
        float sdt = dt / subSteps;

        for (int step = 0; step < subSteps; ++step) {
            // 1. Predict positions 
            for (auto& p : particles) {
                if (p.invMass == 0.0f) continue;
                p.vel += gravity * sdt;
                p.prevPos = p.pos;
                p.pos += p.vel * sdt;
            }

            // Reset accumulated multipliers at the beginning of each substep sub-iteration loop
            for (auto& c : constraints) {
                c.lambda = 0.0f;
            }

            // 2. Solve Internal Distance Constraints (True XPBD)
            for (auto& c : constraints) {
                SoftBodyParticle& p0 = particles[c.id0];
                SoftBodyParticle& p1 = particles[c.id1];

                float w0 = p0.invMass;
                float w1 = p1.invMass;
                float wSum = w0 + w1;
                if (wSum == 0.0f) continue;

                glm::vec3 dir = p0.pos - p1.pos;
                float len = glm::length(dir);
                if (len < 0.0001f) continue;

                dir /= len; // Faster normalization
                float C = len - c.restLength;

                // XPBD Compliance Evaluation
                float tildeCompliance = c.compliance / (sdt * sdt);
                float denominator = wSum + tildeCompliance;
                if (denominator < 0.0001f) continue;

                // Calculate the true change in lambda, factoring in the constraint history
                float dLambda = (-C - tildeCompliance * c.lambda) / denominator;
                c.lambda += dLambda;

                if (w0 > 0.0f) p0.pos += w0 * dLambda * dir;
                if (w1 > 0.0f) p1.pos -= w1 * dLambda * dir;
            }

            // 3. Resolve Environment Collisions with Stable Friction Dampening
            for (auto& p : particles) {
                if (p.pos.y < 0.0f) {
                    p.pos.y = 0.0f;
                    // Dynamic Slide-Friction instead of completely freezing spatial axes
                    p.pos.x = glm::mix(p.pos.x, p.prevPos.x, 0.2f);
                    p.pos.z = glm::mix(p.pos.z, p.prevPos.z, 0.2f);
                }
            }

            // 4. Velocity Update (Verlet Update rule)
            for (auto& p : particles) {
                if (p.invMass == 0.0f) continue;
                p.vel = (p.pos - p.prevPos) / sdt;
            }
        }

        updateBuffers();
    }

    void Draw(Shader& shader) {
        if (VAO == 0) return;
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

private:

    std::vector<glm::vec2> particleTexCoords;

    void addConstraint(int id0, int id1, float compliance) {
        EdgeConstraint c;
        c.id0 = id0;
        c.id1 = id1;
        c.restLength = glm::distance(particles[id0].pos, particles[id1].pos);
        c.compliance = compliance;
        c.lambda = 0.0f;
        constraints.push_back(c);
    }

    void cleanupBuffers() {
        if (VAO != 0) { glDeleteVertexArrays(1, &VAO); VAO = 0; }
        if (VBO != 0) { glDeleteBuffers(1, &VBO); VBO = 0; }
        if (EBO != 0) { glDeleteBuffers(1, &EBO); EBO = 0; }
    }
    void computeSmoothNormals(std::vector<SoftBodyVertex>& outVertices) {
        // Initialize/reset position properties and clear accumulated normals
        for (size_t i = 0; i < particles.size(); ++i) {
            outVertices[i].Position = particles[i].pos;
            outVertices[i].Normal = glm::vec3(0.0f);
            // Apply mapped texture tracking values if populated
            if (i < particleTexCoords.size()) {
                outVertices[i].TexCoords = particleTexCoords[i];
            }
            else {
                outVertices[i].TexCoords = glm::vec2(0.0f);
            }
        }

        // Accumulate face plane cross-products for perfect smooth deforming normals
        for (size_t i = 0; i < indices.size(); i += 3) {
            unsigned int idx0 = indices[i];
            unsigned int idx1 = indices[i + 1];
            unsigned int idx2 = indices[i + 2];

            glm::vec3 v0 = particles[idx0].pos;
            glm::vec3 v1 = particles[idx1].pos;
            glm::vec3 v2 = particles[idx2].pos;

            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 faceNormal = glm::cross(edge1, edge2);

            outVertices[idx0].Normal += faceNormal;
            outVertices[idx1].Normal += faceNormal;
            outVertices[idx2].Normal += faceNormal;
        }

        // Normalize vectors to make them unit length
        for (size_t i = 0; i < outVertices.size(); ++i) {
            if (glm::length(outVertices[i].Normal) > 0.0001f) {
                outVertices[i].Normal = glm::normalize(outVertices[i].Normal);
            }
        }
    }

    void setupBuffers() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        // Pre-allocate structural vertex layout data array
        std::vector<SoftBodyVertex> vertices(particles.size());
        // Initial normal generation step
        computeSmoothNormals(vertices);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(SoftBodyVertex), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        // Attribute 0: Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SoftBodyVertex), (void*)offsetof(SoftBodyVertex, Position));

        // Attribute 1: Normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SoftBodyVertex), (void*)offsetof(SoftBodyVertex, Normal));

        // Attribute 2: UV Texture Coordinates
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(SoftBodyVertex), (void*)offsetof(SoftBodyVertex, TexCoords));

        glBindVertexArray(0);
    }

    void updateBuffers() {
        if (VBO == 0) return;

        std::vector<SoftBodyVertex> vertices(particles.size());
        computeSmoothNormals(vertices);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(SoftBodyVertex), vertices.data());
    }


};

#endif