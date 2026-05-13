#ifndef GJK_CTA_UTILS_GLSL
#define GJK_CTA_UTILS_GLSL

// GJK and CTA implementations in GLSL

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

// Forward declarations for shapes
vec3 support_sphere(vec3 center, float radius, vec3 dir) {
    float l = length(dir);
    vec3 dir_norm = l > 1e-6 ? dir / l : vec3(1.0, 0.0, 0.0);
    return center + dir_norm * radius;
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
            return do_simplex(simplex, dir);
        } else if (dot(acd, ao) > 0.0) {
            simplex.points[0] = d;
            simplex.points[1] = c;
            simplex.points[2] = a;
            simplex.count = 3;
            dir = acd;
            return do_simplex(simplex, dir);
        } else if (dot(adb, ao) > 0.0) {
            simplex.points[0] = d;
            simplex.points[1] = b;
            simplex.points[2] = a;
            simplex.count = 3;
            dir = adb;
            return do_simplex(simplex, dir);
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
        
        vec3 n = dist > 1e-6 ? normalize(p_a - p_b) : vec3(1.0, 0.0, 0.0);
        float v_closing = -dot(v_rel, n);
        
        if (v_closing <= 0.0) return false;
        
        float delta_t = dist / v_closing;
        t += delta_t;
        
        if (t > 1.0) return false;
    }
    
    return false;
}

#endif // GJK_CTA_UTILS_GLSL