#ifndef GJK_CTA_UTILS_GLSL
#define GJK_CTA_UTILS_GLSL

// GJK and CTA implementations in GLSL
#extension GL_EXT_control_flow_attributes : require

struct MinkowskiPoint {
    vec3 point;
    vec3 point_a;
    vec3 point_b;
};

// Represents a 4-point simplex (tetrahedron)
struct Simplex {
    MinkowskiPoint points[4];
    int count;
};

// Support functions for shapes
vec3 support_sphere(vec3 center, float radius, vec3 dir) {
    float l = length(dir);
    vec3 dir_norm = l > 1e-6 ? dir / l : vec3(1.0, 0.0, 0.0);
    return center + dir_norm * radius;
}

vec3 support_obb(vec3 origin, vec3 x_axis, vec3 y_axis, vec3 z_axis, vec3 extents, vec3 dir) {
    vec3 result = origin;
    result += x_axis * (dot(x_axis, dir) > 0.0 ? extents.x : -extents.x);
    result += y_axis * (dot(y_axis, dir) > 0.0 ? extents.y : -extents.y);
    result += z_axis * (dot(z_axis, dir) > 0.0 ? extents.z : -extents.z);
    return result;
}

// EPA Implementation
struct Face {
    int a, b, c;
    vec3 normal;
    float distance;
};

Face create_face(Simplex polytope, int a, int b, int c) {
    vec3 ab = polytope.points[b].point - polytope.points[a].point;
    vec3 ac = polytope.points[c].point - polytope.points[a].point;
    vec3 n = cross(ab, ac);
    
    vec3 normal = dot(n, n) > 1e-8 ? normalize(n) : vec3(1.0, 0.0, 0.0);
    float d = dot(normal, polytope.points[a].point);
    
    if (d < 0.0) {
        normal = -normal;
        d = -d;
        int temp = b; b = c; c = temp;
    }
    
    Face f;
    f.a = a; f.b = b; f.c = c;
    f.normal = normal;
    f.distance = d;
    return f;
}

Face create_face_poly(MinkowskiPoint points[12], int a, int b, int c) {
    vec3 ab = points[b].point - points[a].point;
    vec3 ac = points[c].point - points[a].point;
    vec3 n = cross(ab, ac);
    
    vec3 normal = dot(n, n) > 1e-8 ? normalize(n) : vec3(1.0, 0.0, 0.0);
    float d = dot(normal, points[a].point);
    
    if (d < 0.0) {
        normal = -normal;
        d = -d;
        int temp = b; b = c; c = temp;
    }
    
    Face f;
    f.a = a; f.b = b; f.c = c;
    f.normal = normal;
    f.distance = d;
    return f;
}

vec3 support_shape(uint shape_type, vec3 shape_data, mat4 transform, vec3 dir);

void epa_distance(inout Simplex simplex, uint type1, vec3 data1, mat4 trans1, uint type2, vec3 data2, mat4 trans2, out float dist, out vec3 p_a, out vec3 p_b, out vec3 epa_normal) {
    MinkowskiPoint polytope_points[12];
    int polytope_count = simplex.count;
    for(int i = 0; i < simplex.count; i++) {
        polytope_points[i] = simplex.points[i];
    }
    
    Face faces[16];
    int num_faces = 0;
    
    if (polytope_count == 4) {
        faces[num_faces++] = create_face_poly(polytope_points, 0, 1, 2);
        faces[num_faces++] = create_face_poly(polytope_points, 0, 3, 1);
        faces[num_faces++] = create_face_poly(polytope_points, 0, 2, 3);
        faces[num_faces++] = create_face_poly(polytope_points, 1, 3, 2);
    } else {
        dist = 0.0; p_a = vec3(0.0); p_b = vec3(0.0);
        return;
    }
    
    [[dont_unroll]]
    for (int iter = 0; iter < 16; ++iter) {
        int closest_face_idx = 0;
        float min_dist = faces[0].distance;
        for (int i = 1; i < num_faces; ++i) {
            if (faces[i].distance < min_dist) {
                min_dist = faces[i].distance;
                closest_face_idx = i;
            }
        }
        
        Face closest_face = faces[closest_face_idx];
        vec3 search_dir = closest_face.normal;
        
        vec3 supp_a = support_shape(type1, data1, trans1, search_dir);
        vec3 supp_b = support_shape(type2, data2, trans2, -search_dir);
        vec3 new_pt = supp_a - supp_b;
        
        float d = dot(new_pt, search_dir);
        if (d - min_dist < 1e-4) {
            MinkowskiPoint a = polytope_points[closest_face.a];
            MinkowskiPoint b = polytope_points[closest_face.b];
            MinkowskiPoint c = polytope_points[closest_face.c];
            
            vec3 n = closest_face.normal;
            vec3 p = n * min_dist;
            
            vec3 v0 = b.point - a.point;
            vec3 v1 = c.point - a.point;
            vec3 v2 = p - a.point;
            
            float d00 = dot(v0, v0);
            float d01 = dot(v0, v1);
            float d11 = dot(v1, v1);
            float d20 = dot(v2, v0);
            float d21 = dot(v2, v1);
            
            float denom = d00 * d11 - d01 * d01;
            float v = 0.333, w = 0.333;
            if (abs(denom) >= 1e-6) {
                v = (d11 * d20 - d01 * d21) / denom;
                w = (d00 * d21 - d01 * d20) / denom;
            }
            float u = 1.0 - v - w;
            
            dist = -min_dist;
            epa_normal = closest_face.normal;
            p_a = a.point_a * u + b.point_a * v + c.point_a * w;
            p_b = a.point_b * u + b.point_b * v + c.point_b * w;
            return;
        }
        
        if (polytope_count >= 12) break; // Maximum vertices
        
        MinkowskiPoint mp;
        mp.point = new_pt;
        mp.point_a = supp_a;
        mp.point_b = supp_b;
        int new_idx = polytope_count;
        polytope_points[polytope_count++] = mp;
        
        ivec2 edges[24];
        int num_edges = 0;
        
        int i = 0;
        while (i < num_faces) {
            if (dot(faces[i].normal, new_pt - polytope_points[faces[i].a].point) > 0.0) {
                Face f = faces[i];
                faces[i] = faces[--num_faces]; // Remove face
                
                // Add edges, removing pairs
                int e[6] = {f.a, f.b, f.b, f.c, f.c, f.a};
                [[dont_unroll]]
                for (int j = 0; j < 3; ++j) {
                    int ea = e[j*2];
                    int eb = e[j*2+1];
                    bool found = false;
                    [[dont_unroll]]
                    for (int k = 0; k < num_edges; ++k) {
                        if (edges[k].x == eb && edges[k].y == ea) {
                            edges[k] = edges[--num_edges];
                            found = true;
                            break;
                        }
                    }
                    if (!found && num_edges < 24) {
                        edges[num_edges++] = ivec2(ea, eb);
                    }
                }
            } else {
                i++;
            }
        }
        
        if (num_edges == 0) break;
        
        [[dont_unroll]]
        for (int k = 0; k < num_edges; ++k) {
            if (num_faces < 16) {
                faces[num_faces++] = create_face_poly(polytope_points, edges[k].x, edges[k].y, new_idx);
            }
        }
    }
    
    int best_face = 0;
    float min_dist_final = faces[0].distance;
    [[dont_unroll]]
    for (int i = 1; i < num_faces; ++i) {
        if (faces[i].distance < min_dist_final) {
            min_dist_final = faces[i].distance;
            best_face = i;
        }
    }
    
    Face closest_face_final = faces[best_face];
    MinkowskiPoint a = polytope_points[closest_face_final.a];
    MinkowskiPoint b = polytope_points[closest_face_final.b];
    MinkowskiPoint c = polytope_points[closest_face_final.c];
    
    vec3 n = closest_face_final.normal;
    vec3 p = n * min_dist_final;
    
    vec3 v0 = b.point - a.point;
    vec3 v1 = c.point - a.point;
    vec3 v2 = p - a.point;
    
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    
    float denom = d00 * d11 - d01 * d01;
    float v = 0.333, w = 0.333;
    if (abs(denom) >= 1e-6) {
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
    }
    float u = 1.0 - v - w;
    
    dist = -min_dist_final;
    epa_normal = closest_face_final.normal;
    p_a = a.point_a * u + b.point_a * v + c.point_a * w;
    p_b = a.point_b * u + b.point_b * v + c.point_b * w;
}


// Do-Simplex implementation
bool do_simplex(inout Simplex simplex, inout vec3 dir) {
    if (simplex.count == 2) {
        MinkowskiPoint a = simplex.points[1];
        MinkowskiPoint b = simplex.points[0];
        vec3 ab = b.point - a.point;
        vec3 ao = -a.point;
        
        if (dot(ab, ao) > 0.0) {
            dir = cross(cross(ab, ao), ab);
        } else {
            simplex.points[0] = a;
            simplex.count = 1;
            dir = ao;
        }
        return false;
    } else if (simplex.count == 3) {
        MinkowskiPoint a = simplex.points[2];
        MinkowskiPoint b = simplex.points[1];
        MinkowskiPoint c = simplex.points[0];

        vec3 ab = b.point - a.point;
        vec3 ac = c.point - a.point;
        vec3 ao = -a.point;
        vec3 abc = cross(ab, ac);

        if (dot(cross(abc, ac), ao) > 0.0) {
            if (dot(ac, ao) > 0.0) {
                simplex.points[0] = c;
                simplex.points[1] = a;
                simplex.count = 2;
                dir = cross(cross(ac, ao), ac);
            } else {
                if (dot(ab, ao) > 0.0) {
                    simplex.points[0] = b;
                    simplex.points[1] = a;
                    simplex.count = 2;
                    dir = cross(cross(ab, ao), ab);
                } else {
                    simplex.points[0] = a;
                    simplex.count = 1;
                    dir = ao;
                }
            }
        } else {
            if (dot(cross(ab, abc), ao) > 0.0) {
                if (dot(ab, ao) > 0.0) {
                    simplex.points[0] = b;
                    simplex.points[1] = a;
                    simplex.count = 2;
                    dir = cross(cross(ab, ao), ab);
                } else {
                    simplex.points[0] = a;
                    simplex.count = 1;
                    dir = ao;
                }
            } else {
                if (dot(abc, ao) > 0.0) {
                    dir = abc;
                } else {
                    // Swap B and C to maintain winding
                    MinkowskiPoint temp = simplex.points[0];
                    simplex.points[0] = simplex.points[1];
                    simplex.points[1] = temp;
                    dir = -abc;
                }
            }
        }
        return false;
    } else if (simplex.count == 4) {
        MinkowskiPoint a = simplex.points[3];
        MinkowskiPoint b = simplex.points[2];
        MinkowskiPoint c = simplex.points[1];
        MinkowskiPoint d = simplex.points[0];

        vec3 ab = b.point - a.point;
        vec3 ac = c.point - a.point;
        vec3 ad = d.point - a.point;
        vec3 ao = -a.point;

        vec3 abc = cross(ab, ac);
        vec3 acd = cross(ac, ad);
        vec3 adb = cross(ad, ab);

        if (dot(abc, ao) > 0.0) {
            simplex.points[0] = c;
            simplex.points[1] = b;
            simplex.points[2] = a;
            simplex.count = 3;
            dir = abc;
            return false;
        } else if (dot(acd, ao) > 0.0) {
            simplex.points[0] = d;
            simplex.points[1] = c;
            simplex.points[2] = a;
            simplex.count = 3;
            dir = acd;
            return false;
        } else if (dot(adb, ao) > 0.0) {
            simplex.points[0] = d;
            simplex.points[1] = b;
            simplex.points[2] = a;
            simplex.count = 3;
            dir = adb;
            return false;
        } else {
            return true; // Origin is inside the tetrahedron
        }
    }
    return false;
}

void compute_closest_points(in Simplex simplex, out vec3 closest_a, out vec3 closest_b) {
    if (simplex.count == 1) {
        closest_a = simplex.points[0].point_a;
        closest_b = simplex.points[0].point_b;
    } else if (simplex.count == 2) {
        MinkowskiPoint a = simplex.points[1];
        MinkowskiPoint b = simplex.points[0];
        
        vec3 ab = b.point - a.point;
        float l2 = dot(ab, ab);
        float t = l2 > 1e-6 ? clamp(dot(-a.point, ab) / l2, 0.0, 1.0) : 0.0;
        
        closest_a = a.point_a + (b.point_a - a.point_a) * t;
        closest_b = a.point_b + (b.point_b - a.point_b) * t;
    } else if (simplex.count == 3) {
        MinkowskiPoint a = simplex.points[2];
        MinkowskiPoint b = simplex.points[1];
        MinkowskiPoint c = simplex.points[0];

        vec3 ab = b.point - a.point;
        vec3 ac = c.point - a.point;
        
        vec3 n = cross(ab, ac);
        float n_len_sq = dot(n, n);
        
        if (n_len_sq < 1e-6) {
             closest_a = a.point_a;
             closest_b = a.point_b;
             return;
        }
        
        vec3 ao = -a.point;
        float u = dot(cross(ac, n), ao) / n_len_sq;
        float v = dot(cross(n, ab), ao) / n_len_sq;
        float w = 1.0 - u - v;
        
        closest_a = a.point_a * w + b.point_a * u + c.point_a * v;
        closest_b = a.point_b * w + b.point_b * u + c.point_b * v;
    } else {
        closest_a = vec3(0.0);
        closest_b = vec3(0.0);
    }
}

// Distance between two generic shapes
vec3 support_shape(uint shape_type, vec3 shape_data, mat4 transform, vec3 dir) {
    vec3 local_dir = (inverse(transform) * vec4(dir, 0.0)).xyz;
    vec3 result = vec3(0.0);
    
    if (shape_type == 2) { // Sphere
        float radius = shape_data.x;
        float l = length(local_dir);
        vec3 dir_norm = l > 1e-6 ? local_dir / l : vec3(1.0, 0.0, 0.0);
        result = dir_norm * radius;
    } else if (shape_type == 1) { // OBB
        vec3 extents = shape_data;
        result.x = dot(vec3(1,0,0), local_dir) > 0.0 ? extents.x : -extents.x;
        result.y = dot(vec3(0,1,0), local_dir) > 0.0 ? extents.y : -extents.y;
        result.z = dot(vec3(0,0,1), local_dir) > 0.0 ? extents.z : -extents.z;
    }
    
    return (transform * vec4(result, 1.0)).xyz;
}

float gjk_distance_generic(uint type1, vec3 data1, mat4 trans1, uint type2, vec3 data2, mat4 trans2, out vec3 p_a, out vec3 p_b, out vec3 epa_normal) {
    vec3 dir = normalize(vec3(1.0, 0.1, 0.01));
    
    vec3 support_a = support_shape(type1, data1, trans1, -dir);
    vec3 support_b = support_shape(type2, data2, trans2, dir);
    
    Simplex simplex;
    simplex.points[0].point = support_a - support_b;
    simplex.points[0].point_a = support_a;
    simplex.points[0].point_b = support_b;
    simplex.count = 1;
    
    vec3 v = simplex.points[0].point;
    vec3 last_valid_dir = dir;
    
    [[dont_unroll]]
    for (int i = 0; i < 64; ++i) {
        if (dot(v, v) < 1e-6) {
            compute_closest_points(simplex, p_a, p_b);
            epa_normal = -last_valid_dir;
            return 0.0;
        }
        
        dir = -v;
        last_valid_dir = dir;
        vec3 p1 = support_shape(type1, data1, trans1, dir);
        vec3 p2 = support_shape(type2, data2, trans2, -dir);
        
        MinkowskiPoint w;
        w.point = p1 - p2;
        w.point_a = p1;
        w.point_b = p2;
        
        float v_dot_dir = dot(v, dir);
        float w_dot_dir = dot(w.point, dir);
        if (w_dot_dir - v_dot_dir < 1e-4) {
            compute_closest_points(simplex, p_a, p_b);
            epa_normal = -last_valid_dir;
            return length(p_a - p_b);
        }
        
        simplex.points[simplex.count] = w;
        simplex.count++;
        
        if (do_simplex(simplex, dir)) {
            // Origin is enclosed, they intersect! Use EPA.
            float dist_out;
            epa_distance(simplex, type1, data1, trans1, type2, data2, trans2, dist_out, p_a, p_b, epa_normal);
            return dist_out;
        }
        
        if (simplex.count == 1) { v = simplex.points[0].point; }
        else if (simplex.count == 2) { 
            compute_closest_points(simplex, p_a, p_b);
            v = p_a - p_b;
        }
        else if (simplex.count == 3) {
            compute_closest_points(simplex, p_a, p_b);
            v = p_a - p_b;
        }
    }
    
    compute_closest_points(simplex, p_a, p_b);
    epa_normal = -last_valid_dir; // No overlap, fallback normal
    return length(p_a - p_b);
}

// Distance between two spheres
float gjk_distance_spheres(vec3 c1, float r1, vec3 c2, float r2, out vec3 p_a, out vec3 p_b) {
    vec3 dir = vec3(1.0, 0.0, 0.0);
    
    vec3 support_a = support_sphere(c1, r1, -dir);
    vec3 support_b = support_sphere(c2, r2, dir);
    
    Simplex simplex;
    simplex.points[0].point = support_a - support_b;
    simplex.points[0].point_a = support_a;
    simplex.points[0].point_b = support_b;
    simplex.count = 1;
    
    vec3 v = simplex.points[0].point;
    
    const int MAX_ITERATIONS = 64;
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        if (dot(v, v) < 1e-6) break;
        
        dir = -v;
        vec3 p1 = support_sphere(c1, r1, dir);
        vec3 p2 = support_sphere(c2, r2, -dir);
        
        MinkowskiPoint w;
        w.point = p1 - p2;
        w.point_a = p1;
        w.point_b = p2;
        
        float v_dot_dir = dot(v, dir);
        float w_dot_dir = dot(w.point, dir);
        if (w_dot_dir - v_dot_dir < 1e-4) break;
        
        simplex.points[simplex.count] = w;
        simplex.count++;
        
        if (do_simplex(simplex, dir)) {
            p_a = vec3(0.0); p_b = vec3(0.0);
            return 0.0;
        }
        
        if (simplex.count == 1) { v = simplex.points[0].point; }
        else if (simplex.count == 2) { 
            compute_closest_points(simplex, p_a, p_b);
            v = p_a - p_b;
        }
        else if (simplex.count == 3) {
            compute_closest_points(simplex, p_a, p_b);
            v = p_a - p_b;
        }
    }
    
    compute_closest_points(simplex, p_a, p_b);
    return length(p_a - p_b);
}

bool compute_toi_spheres(vec3 c1, float r1, vec3 v1, vec3 c2, float r2, vec3 v2, float time_tolerance, int max_iterations, out float out_toi) {
    float t = 0.0;
    
    vec3 v_rel = v1 - v2;
    float v_rel_max = length(v_rel);
    
    if (v_rel_max < 1e-6) {
        vec3 p_a, p_b;
        float dist = gjk_distance_spheres(c1, r1, c2, r2, p_a, p_b);
        if (dist < time_tolerance) {
            out_toi = 0.0;
            return true;
        }
        return false;
    }
    
    for (int i = 0; i < max_iterations; ++i) {
        vec3 cur_c1 = c1 + v1 * t;
        vec3 cur_c2 = c2 + v2 * t;
        
        vec3 p_a, p_b;
        float dist = gjk_distance_spheres(cur_c1, r1, cur_c2, r2, p_a, p_b);
        
        if (dist < time_tolerance) {
            out_toi = t;
            return true;
        }

        vec3 n_dir = dist > 1e-6 ? normalize(p_a - p_b) : vec3(1.0, 0.0, 0.0);
        float v_closing = -dot(v_rel, n_dir);
        
        if (v_closing <= 0.0) return false;
        
        float delta_t = dist / v_closing;
        t += delta_t;
        
        if (t > 1.0) return false;
    }
    
    return false;
}

bool compute_toi_generic(
    uint type1, vec3 data1, mat4 trans1_start, vec3 v1,
    uint type2, vec3 data2, mat4 trans2_start, vec3 v2,
    float time_tolerance, int max_iterations, out float out_toi,
    out vec3 out_normal, out vec3 out_contact_point, out float out_depth
) {
    float t = 0.0;
    
    vec3 p_a_init, p_b_init, epa_n_init;
    float dist = gjk_distance_generic(type1, data1, trans1_start, type2, data2, trans2_start, p_a_init, p_b_init, epa_n_init);
    
    vec3 v_rel = v1 - v2;
    float v_rel_max = length(v_rel);
    
    if (v_rel_max <= 1e-6) {
        if (dist <= 0.0) {
            out_toi = 0.0;
            out_depth = dist < 0.0 ? -dist : 0.0;
            // EPA normal points from A to B.
            float epa_len = length(epa_n_init);
            out_normal = epa_len > 1e-6 ? epa_n_init / epa_len : vec3(1.0, 0.0, 0.0);
            out_contact_point = (p_a_init + p_b_init) * 0.5;
            return true;
        }
        return false;
    }
    
    vec3 last_valid_normal = v_rel_max > 1e-6 ? normalize(v_rel) : vec3(0.0, -1.0, 0.0);
    
    for (int i = 0; i < max_iterations; ++i) {
        mat4 cur_trans1 = trans1_start;
        cur_trans1[3].xyz += v1 * t;
        
        mat4 cur_trans2 = trans2_start;
        cur_trans2[3].xyz += v2 * t;
        
        vec3 p_a, p_b, epa_n;
        float dist = gjk_distance_generic(type1, data1, cur_trans1, type2, data2, cur_trans2, p_a, p_b, epa_n);
        
        if (dist <= time_tolerance) {
            out_toi = t;
            out_depth = dist < 0.0 ? -dist : 0.0;
            if (dist < 0.0 && length(epa_n) > 1e-6) {
                out_normal = normalize(epa_n);
            } else {
                vec3 n = p_b - p_a;
                float n_len = length(n);
                if (n_len > 1e-6) {
                    out_normal = n / n_len;
                } else {
                    out_normal = last_valid_normal;
                }
            }
            out_contact_point = (p_a + p_b) * 0.5;
            return true;
        }

        vec3 n_dir = dist > 1e-6 ? normalize(p_b - p_a) : last_valid_normal;
        last_valid_normal = n_dir;
        
        float v_closing = -dot(v_rel, n_dir);
        
        if (v_closing <= 0.0) return false;
        
        float delta_t = dist / v_closing;
        t += delta_t;
        
        if (t > 1.0) return false;
    }
    
    return false;
}

#endif // GJK_CTA_UTILS_GLSL