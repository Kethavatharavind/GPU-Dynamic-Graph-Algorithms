#ifndef TRIANGLE_CU
#define TRIANGLE_CU

#include "cpu_algo.h"
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif
#define cuda_check CUDA_CHECK


__device__ inline int dev_bsearch(const int* col, int lo, int hi, int key) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int val = col[mid];
        if (val < key) {
            lo = mid + 1;
        } else if (val > key) {
            hi = mid;
        } else {
            return 1;
        }
    }
    return 0;
}

__device__ inline int dev_find_edge(const int* eu, const int* ev, int k, int u, int v) {
    int lo = 0, hi = k;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int mu = eu[mid];
        int mv = ev[mid];
        if (mu < u || (mu == u && mv < v)) {
            lo = mid + 1;
        } else if (mu > u || (mu == u && mv > v)) {
            hi = mid;
        } else {
            return mid;
        }
    }
    return -1;
}

__device__ inline bool dev_deg_less(int a, int deg_a, int b, int deg_b) {
    return (deg_a < deg_b) || (deg_a == deg_b && a < b);
}

// static triangle counting kernel
__global__ void tc_static_warp_kernel(const int* row,
                                      const int* col,
                                      const int* eu,
                                      const int* ev,
                                      int num_edges,
                                      unsigned long long* d_total) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long warp_id = tid >> 5;
    int lane = (int)(tid & 31);
    if (warp_id >= num_edges) return;

    int u = eu[warp_id];
    int v = ev[warp_id];
    int u_lo = row[u], u_hi = row[u + 1];
    int v_lo = row[v], v_hi = row[v + 1];
    int deg_v = v_hi - v_lo;

    unsigned long long cnt = 0;
    for (int i = u_lo + lane; i < u_hi; i += 32) {
        int w = col[i];
        int deg_w = row[w + 1] - row[w];
        if (dev_deg_less(v, deg_v, w, deg_w)) {
            cnt += (unsigned long long)dev_bsearch(col, v_lo, v_hi, w);
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        cnt += __shfl_down_sync(0xffffffff, cnt, off);
    }
    if (lane == 0 && cnt > 0) {
        atomicAdd(d_total, cnt);
    }
}

// batch update kernel for triangle counting
__global__ void tc_batch_delta_kernel(const int* row,
                                      const int* col,
                                      const int* upd_u,
                                      const int* upd_v,
                                      int k,
                                      unsigned long long* d_delta) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long warp_id = tid >> 5;
    int lane = (int)(tid & 31);
    if (warp_id >= k) return;

    int u = upd_u[warp_id];
    int v = upd_v[warp_id];
    int u_lo = row[u], u_hi = row[u + 1];
    int v_lo = row[v], v_hi = row[v + 1];

    int a_lo = u_lo, a_hi = u_hi;
    int b_lo = v_lo, b_hi = v_hi;
    int b = v;
    if ((v_hi - v_lo) < (u_hi - u_lo)) {
        a_lo = v_lo;
        a_hi = v_hi;
        b_lo = u_lo;
        b_hi = u_hi;
        b = u;
    }

    unsigned long long cnt = 0;
    for (int i = a_lo + lane; i < a_hi; i += 32) {
        int w = col[i];
        if (w == b) continue;
        if (dev_bsearch(col, b_lo, b_hi, w)) {
            int u1 = (u < w) ? u : w;
            int v1 = (u < w) ? w : u;
            int p1 = dev_find_edge(upd_u, upd_v, k, u1, v1);
            if (p1 != -1 && p1 < warp_id) continue;

            int u2 = (v < w) ? v : w;
            int v2 = (v < w) ? w : v;
            int p2 = dev_find_edge(upd_u, upd_v, k, u2, v2);
            if (p2 != -1 && p2 < warp_id) continue;

            cnt++;
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        cnt += __shfl_down_sync(0xffffffff, cnt, off);
    }
    if (lane == 0 && cnt > 0) {
        atomicAdd(d_delta, cnt);
    }
}

struct gpu_tc_state {
    int* d_row = nullptr;
    int* d_col = nullptr;
    int* d_eu = nullptr;
    int* d_ev = nullptr;
    int* d_upd_u = nullptr;
    int* d_upd_v = nullptr;
    unsigned long long* d_cnt = nullptr;

    int cap_n = 0;
    int cap_m = 0;
    int cap_e = 0;
    int cap_upd = 0;
    size_t mem_bytes = 0;

    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;

    void init_events() {
        if (!ev0) cuda_check(cudaEventCreate(&ev0));
        if (!ev1) cuda_check(cudaEventCreate(&ev1));
        if (!d_cnt) {
            cuda_check(cudaMalloc(&d_cnt, sizeof(unsigned long long)));
            mem_bytes += sizeof(unsigned long long);
        }
    }

    void ensure_csr_cap(int n, int m) {
        int need_n = std::max(1, n + 1);
        int need_m = std::max(1, m);
        if (need_n > cap_n) {
            if (d_row) {
                cudaFree(d_row);
                mem_bytes -= (size_t)cap_n * sizeof(int);
            }
            cap_n = need_n;
            cuda_check(cudaMalloc(&d_row, (size_t)cap_n * sizeof(int)));
            mem_bytes += (size_t)cap_n * sizeof(int);
        }
        if (need_m > cap_m) {
            if (d_col) {
                cudaFree(d_col);
                mem_bytes -= (size_t)cap_m * sizeof(int);
            }
            int dbl_m = (cap_m <= 1000000000) ? (cap_m * 2) : need_m;
            cap_m = std::max(need_m, dbl_m);
            cuda_check(cudaMalloc(&d_col, (size_t)cap_m * sizeof(int)));
            mem_bytes += (size_t)cap_m * sizeof(int);
        }
    }

    void ensure_edge_cap(int num_edges) {
        int need_e = std::max(1, num_edges);
        if (need_e > cap_e) {
            if (d_eu) {
                cudaFree(d_eu);
                cudaFree(d_ev);
                mem_bytes -= 2ULL * (size_t)cap_e * sizeof(int);
            }
            int dbl_e = (cap_e <= 1000000000) ? (cap_e * 2) : need_e;
            cap_e = std::max(need_e, dbl_e);
            cuda_check(cudaMalloc(&d_eu, (size_t)cap_e * sizeof(int)));
            cuda_check(cudaMalloc(&d_ev, (size_t)cap_e * sizeof(int)));
            mem_bytes += 2ULL * (size_t)cap_e * sizeof(int);
        }
    }

    void ensure_upd_cap(int k) {
        int need_k = std::max(1, k);
        if (need_k > cap_upd) {
            if (d_upd_u) {
                cudaFree(d_upd_u);
                cudaFree(d_upd_v);
                mem_bytes -= 2ULL * (size_t)cap_upd * sizeof(int);
            }
            int dbl_u = (cap_upd <= 1000000000) ? (cap_upd * 2) : need_k;
            cap_upd = std::max(need_k, dbl_u);
            cuda_check(cudaMalloc(&d_upd_u, (size_t)cap_upd * sizeof(int)));
            cuda_check(cudaMalloc(&d_upd_v, (size_t)cap_upd * sizeof(int)));
            mem_bytes += 2ULL * (size_t)cap_upd * sizeof(int);
        }
    }

    void upload_csr(const csr_graph& csr) {
        ensure_csr_cap(csr.n, csr.m);
        if (csr.n >= 0) {
            cuda_check(cudaMemcpy(d_row, csr.row.data(),
                                  (size_t)(csr.n + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
        if (csr.m > 0) {
            cuda_check(cudaMemcpy(d_col, csr.col.data(),
                                  (size_t)csr.m * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
    }

    long long static_count(const csr_graph& csr, double& kernel_ms, double& e2e_ms) {
        kernel_ms = 0.0;
        e2e_ms = 0.0;
        if (csr.n == 0 || csr.m == 0) {
            init_events();
            upload_csr(csr);
            return 0;
        }

        my_timer t_e2e;
        t_e2e.start();
        init_events();

        int num_edges = csr.m / 2;
        std::vector<int> eu(num_edges), ev(num_edges);
        int idx = 0;
        for (int u = 0; u < csr.n; ++u) {
            for (int i = csr.row[u]; i < csr.row[u + 1]; ++i) {
                int v = csr.col[i];
                if (deg_less(u, csr.deg[u], v, csr.deg[v])) {
                    eu[idx] = u;
                    ev[idx] = v;
                    idx++;
                }
            }
        }

        upload_csr(csr);
        ensure_edge_cap(num_edges);
        cuda_check(cudaMemcpy(d_eu, eu.data(), (size_t)num_edges * sizeof(int), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(d_ev, ev.data(), (size_t)num_edges * sizeof(int), cudaMemcpyHostToDevice));
        cuda_check(cudaMemset(d_cnt, 0, sizeof(unsigned long long)));

        int threads = 256;
        int warps_per_blk = threads >> 5;
        int blocks = (num_edges + warps_per_blk - 1) / warps_per_blk;

        cuda_check(cudaEventRecord(ev0));
        tc_static_warp_kernel<<<blocks, threads>>>(d_row, d_col, d_eu, d_ev, num_edges, d_cnt);
        cuda_check(cudaGetLastError());
        cuda_check(cudaEventRecord(ev1));
        cuda_check(cudaEventSynchronize(ev1));

        float kms = 0.0f;
        cuda_check(cudaEventElapsedTime(&kms, ev0, ev1));
        kernel_ms = (double)kms;

        unsigned long long ans = 0;
        cuda_check(cudaMemcpy(&ans, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        e2e_ms = t_e2e.stop_ms();
        return (long long)ans;
    }

    unsigned long long run_batch_delta(const std::vector<edge_pair>& edges, double& kernel_ms) {
        if (edges.empty()) return 0;
        int k = (int)edges.size();
        std::vector<int> hu(k), hv(k);
        for (int i = 0; i < k; ++i) {
            hu[i] = edges[i].u;
            hv[i] = edges[i].v;
        }
        ensure_upd_cap(k);
        cuda_check(cudaMemcpy(d_upd_u, hu.data(), (size_t)k * sizeof(int), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(d_upd_v, hv.data(), (size_t)k * sizeof(int), cudaMemcpyHostToDevice));
        cuda_check(cudaMemset(d_cnt, 0, sizeof(unsigned long long)));

        int threads = 256;
        int warps_per_blk = threads >> 5;
        int blocks = (k + warps_per_blk - 1) / warps_per_blk;

        cuda_check(cudaEventRecord(ev0));
        tc_batch_delta_kernel<<<blocks, threads>>>(d_row, d_col, d_upd_u, d_upd_v, k, d_cnt);
        cuda_check(cudaGetLastError());
        cuda_check(cudaEventRecord(ev1));
        cuda_check(cudaEventSynchronize(ev1));

        float kms = 0.0f;
        cuda_check(cudaEventElapsedTime(&kms, ev0, ev1));
        kernel_ms += (double)kms;

        unsigned long long delta = 0;
        cuda_check(cudaMemcpy(&delta, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        return delta;
    }

    void apply_batch(dyn_graph& g_gpu,
                     const std::vector<edge_op>& batch,
                     long long& tri_cnt,
                     double& kernel_ms,
                     double& e2e_ms) {
        my_timer t_e2e;
        t_e2e.start();

        if (!d_row) {
            init_events();
            upload_csr(g_gpu.to_csr());
        }

        std::vector<edge_pair> net_del, net_ins;
        resolve_batch(g_gpu, batch, net_del, net_ins);
        if (net_del.empty() && net_ins.empty()) {
            e2e_ms += t_e2e.stop_ms();
            return;
        }

        // Count triangles destroyed in current graph g0 before removing net_del
        unsigned long long destroyed = run_batch_delta(net_del, kernel_ms);

        // Update host dynamic graph and sync CSR to GPU
        for (size_t i = 0; i < net_del.size(); ++i) {
            g_gpu.del_edge(net_del[i].u, net_del[i].v);
        }
        for (size_t i = 0; i < net_ins.size(); ++i) {
            g_gpu.add_edge(net_ins[i].u, net_ins[i].v);
        }
        csr_graph csr1 = g_gpu.to_csr();
        upload_csr(csr1);

        // Count triangles created in updated graph g1
        unsigned long long created = run_batch_delta(net_ins, kernel_ms);

        tri_cnt = tri_cnt + (long long)created - (long long)destroyed;
        e2e_ms += t_e2e.stop_ms();
    }

    void free_all() {
        if (d_row) cudaFree(d_row);
        if (d_col) cudaFree(d_col);
        if (d_eu) cudaFree(d_eu);
        if (d_ev) cudaFree(d_ev);
        if (d_upd_u) cudaFree(d_upd_u);
        if (d_upd_v) cudaFree(d_upd_v);
        if (d_cnt) cudaFree(d_cnt);
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        d_row = d_col = d_eu = d_ev = d_upd_u = d_upd_v = nullptr;
        d_cnt = nullptr;
        ev0 = ev1 = nullptr;
        cap_n = cap_m = cap_e = cap_upd = 0;
        mem_bytes = 0;
    }
};

#endif

