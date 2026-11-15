#ifndef COMPONENTS_CUH
#define COMPONENTS_CUH

#include "cpu_algo.h"
#include "cuda_util.cuh"

__global__ void cc_init_kernel(int* comp, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) comp[u] = u;
}

__global__ void cc_mark_affected_kernel(const int* comp,
                                          const int* del_u,
                                          const int* del_v,
                                          int k_del,
                                          int* affected_root) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k_del) return;
    int ru = comp[del_u[idx]];
    int rv = comp[del_v[idx]];
    affected_root[ru] = 1;
    affected_root[rv] = 1;
}

__global__ void cc_reset_affected_kernel(int* comp,
                                           const int* affected_root,
                                           int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    if (affected_root[comp[u]]) {
        comp[u] = u;
    }
}

__global__ void cc_hook_csr_kernel(const int* row,
                                   const int* col,
                                   int* comp,
                                   int* changed,
                                   int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    int cu = comp[u];
    int lo_idx = row[u];
    int hi_idx = row[u + 1];
    for (int i = lo_idx; i < hi_idx; ++i) {
        int v = col[i];
        int cv = comp[v];
        if (cu != cv) {
            int hi = (cu > cv) ? cu : cv;
            int lo = (cu < cv) ? cu : cv;
            atomicMin(&comp[hi], lo);
            *changed = 1;
        }
    }
}

__global__ void cc_hook_edges_kernel(const int* eu,
                                     const int* ev,
                                     int k,
                                     int* comp,
                                     int* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;

    int cu = comp[eu[idx]];
    int cv = comp[ev[idx]];
    if (cu != cv) {
        int hi = (cu > cv) ? cu : cv;
        int lo = (cu < cv) ? cu : cv;
        atomicMin(&comp[hi], lo);
        *changed = 1;
    }
}

__global__ void cc_jump_kernel(const int* comp_in,
                               int* comp_out,
                               int* changed,
                               int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    int p = comp_in[u];
    int gp = comp_in[p];
    comp_out[u] = gp;
    if (gp != p) {
        *changed = 1;
    }
}

struct gpu_cc_state {
    int* d_row = nullptr;
    int* d_col = nullptr;
    int* d_comp = nullptr;
    int* d_tmp = nullptr;
    int* d_affected = nullptr;
    int* d_upd_u = nullptr;
    int* d_upd_v = nullptr;
    int* d_changed = nullptr;

    int cap_n = 0;
    int cap_m = 0;
    int cap_upd = 0;
    size_t mem_bytes = 0;

    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;

    void init_events() {
        if (!ev0) cuda_check(cudaEventCreate(&ev0));
        if (!ev1) cuda_check(cudaEventCreate(&ev1));
        if (!d_changed) {
            cuda_check(cudaMalloc(&d_changed, sizeof(int)));
            mem_bytes += sizeof(int);
        }
    }

    void ensure_vertex_cap(int n) {
        int need_n = std::max(1, n);
        if (need_n > cap_n) {
            if (d_row) {
                cudaFree(d_row);
                cudaFree(d_comp);
                cudaFree(d_tmp);
                cudaFree(d_affected);
                mem_bytes -= (size_t)(cap_n + 1 + 3 * cap_n) * sizeof(int);
            }
            cap_n = need_n;
            cuda_check(cudaMalloc(&d_row, (size_t)(cap_n + 1) * sizeof(int)));
            cuda_check(cudaMalloc(&d_comp, (size_t)cap_n * sizeof(int)));
            cuda_check(cudaMalloc(&d_tmp, (size_t)cap_n * sizeof(int)));
            cuda_check(cudaMalloc(&d_affected, (size_t)cap_n * sizeof(int)));
            mem_bytes += (size_t)(cap_n + 1 + 3 * cap_n) * sizeof(int);
        }
    }

    void ensure_edge_cap(int m) {
        int need_m = std::max(1, m);
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
        ensure_vertex_cap(csr.n);
        ensure_edge_cap(csr.m);
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

    void flatten_comp(int n, double& kernel_ms) {
        if (n <= 0) return;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        int jump_changed = 1;
        while (jump_changed) {
            jump_changed = 0;
            cuda_check(cudaMemset(d_changed, 0, sizeof(int)));
            cuda_check(cudaEventRecord(ev0));
            cc_jump_kernel<<<blocks, threads>>>(d_comp, d_tmp, d_changed, n);
            cuda_check(cudaGetLastError());
            cuda_check(cudaEventRecord(ev1));
            cuda_check(cudaEventSynchronize(ev1));

            float kms = 0.0f;
            cuda_check(cudaEventElapsedTime(&kms, ev0, ev1));
            kernel_ms += (double)kms;

            std::swap(d_comp, d_tmp);
            cuda_check(cudaMemcpy(&jump_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        }
    }

    void solve_from_csr(int n, int m, double& kernel_ms) {
        if (n <= 0 || m <= 0) return;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        int changed = 1;
        while (changed) {
            changed = 0;
            cuda_check(cudaMemset(d_changed, 0, sizeof(int)));
            cuda_check(cudaEventRecord(ev0));
            cc_hook_csr_kernel<<<blocks, threads>>>(d_row, d_col, d_comp, d_changed, n);
            cuda_check(cudaGetLastError());
            cuda_check(cudaEventRecord(ev1));
            cuda_check(cudaEventSynchronize(ev1));

            float kms = 0.0f;
            cuda_check(cudaEventElapsedTime(&kms, ev0, ev1));
            kernel_ms += (double)kms;

            cuda_check(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            if (changed) {
                flatten_comp(n, kernel_ms);
            }
        }
    }

    int static_cc(const csr_graph& csr,
                  std::vector<int>& comp,
                  double& kernel_ms,
                  double& e2e_ms) {
        kernel_ms = 0.0;
        e2e_ms = 0.0;
        if (csr.n == 0) {
            comp.clear();
            init_events();
            return 0;
        }

        my_timer t_e2e;
        t_e2e.start();
        init_events();
        upload_csr(csr);

        int threads = 256;
        int blocks = (csr.n + threads - 1) / threads;
        cuda_check(cudaEventRecord(ev0));
        cc_init_kernel<<<blocks, threads>>>(d_comp, csr.n);
        cuda_check(cudaGetLastError());
        cuda_check(cudaEventRecord(ev1));
        cuda_check(cudaEventSynchronize(ev1));

        float kms = 0.0f;
        cuda_check(cudaEventElapsedTime(&kms, ev0, ev1));
        kernel_ms += (double)kms;

        solve_from_csr(csr.n, csr.m, kernel_ms);

        comp.resize(csr.n);
        cuda_check(cudaMemcpy(comp.data(), d_comp,
                              (size_t)csr.n * sizeof(int),
                              cudaMemcpyDeviceToHost));
        e2e_ms = t_e2e.stop_ms();
        return count_components(comp);
    }

    int apply_batch(dyn_graph& g_gpu,
                    const std::vector<edge_op>& batch,
                    std::vector<int>& comp,
                    double& kernel_ms,
                    double& e2e_ms) {
        if (g_gpu.n == 0) {
            comp.clear();
            return 0;
        }

        if (!d_comp) {
            double k0 = 0.0, e0 = 0.0;
            static_cc(g_gpu.to_csr(), comp, k0, e0);
            kernel_ms += k0;
            e2e_ms += e0;
        }

        my_timer t_e2e;
        t_e2e.start();

        std::vector<edge_pair> net_del, net_ins;
        resolve_batch(g_gpu, batch, net_del, net_ins);
        if (net_del.empty() && net_ins.empty()) {
            e2e_ms += t_e2e.stop_ms();
            return count_components(comp);
        }

        int n = g_gpu.n;
        if (net_del.empty()) {
            // Insertions only: hook directly on batch edges using existing d_comp
            for (size_t i = 0; i < net_ins.size(); ++i) {
                g_gpu.add_edge(net_ins[i].u, net_ins[i].v);
            }

            int k = (int)net_ins.size();
            std::vector<int> hu(k), hv(k);
            for (int i = 0; i < k; ++i) {
                hu[i] = net_ins[i].u;
                hv[i] = net_ins[i].v;
            }
            ensure_upd_cap(k);
            cuda_check(cudaMemcpy(d_upd_u, hu.data(), (size_t)k * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_upd_v, hv.data(), (size_t)k * sizeof(int), cudaMemcpyHostToDevice));

            int threads = 256;
            int blocks = (k + threads - 1) / threads;
            int changed = 1;
            while (changed) {
                changed = 0;
                cuda_check(cudaMemset(d_changed, 0, sizeof(int)));
                cuda_check(cudaEventRecord(ev0));
                cc_hook_edges_kernel<<<blocks, threads>>>(d_upd_u, d_upd_v, k, d_comp, d_changed);
                cuda_check(cudaGetLastError());
                cuda_check(cudaEventRecord(ev1));
                cuda_check(cudaEventSynchronize(ev1));

                float kms = 0.0f;
                cuda_check(cudaEventElapsedTime(&kms, ev0, ev1));
                kernel_ms += (double)kms;

                cuda_check(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
                if (changed) {
                    flatten_comp(n, kernel_ms);
                }
            }
        } else {
            // Deletions occurred: reset only components touched by deleted edges, then recompute
            int k_del = (int)net_del.size();
            std::vector<int> hu(k_del), hv(k_del);
            for (int i = 0; i < k_del; ++i) {
                hu[i] = net_del[i].u;
                hv[i] = net_del[i].v;
            }
            ensure_upd_cap(k_del);
            cuda_check(cudaMemcpy(d_upd_u, hu.data(), (size_t)k_del * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_upd_v, hv.data(), (size_t)k_del * sizeof(int), cudaMemcpyHostToDevice));
            cuda_check(cudaMemset(d_affected, 0, (size_t)n * sizeof(int)));

            int threads = 256;
            int del_blocks = (k_del + threads - 1) / threads;
            int v_blocks = (n + threads - 1) / threads;

            cuda_check(cudaEventRecord(ev0));
            cc_mark_affected_kernel<<<del_blocks, threads>>>(d_comp, d_upd_u, d_upd_v, k_del, d_affected);
            cuda_check(cudaGetLastError());
            cc_reset_affected_kernel<<<v_blocks, threads>>>(d_comp, d_affected, n);
            cuda_check(cudaGetLastError());
            cuda_check(cudaEventRecord(ev1));
            cuda_check(cudaEventSynchronize(ev1));

            float kms = 0.0f;
            cuda_check(cudaEventElapsedTime(&kms, ev0, ev1));
            kernel_ms += (double)kms;

            for (size_t i = 0; i < net_del.size(); ++i) {
                g_gpu.del_edge(net_del[i].u, net_del[i].v);
            }
            for (size_t i = 0; i < net_ins.size(); ++i) {
                g_gpu.add_edge(net_ins[i].u, net_ins[i].v);
            }
            csr_graph csr1 = g_gpu.to_csr();
            upload_csr(csr1);
            solve_from_csr(csr1.n, csr1.m, kernel_ms);
        }

        comp.resize(n);
        cuda_check(cudaMemcpy(comp.data(), d_comp,
                              (size_t)n * sizeof(int),
                              cudaMemcpyDeviceToHost));
        e2e_ms += t_e2e.stop_ms();
        return count_components(comp);
    }

    void free_all() {
        if (d_row) cudaFree(d_row);
        if (d_col) cudaFree(d_col);
        if (d_comp) cudaFree(d_comp);
        if (d_tmp) cudaFree(d_tmp);
        if (d_affected) cudaFree(d_affected);
        if (d_upd_u) cudaFree(d_upd_u);
        if (d_upd_v) cudaFree(d_upd_v);
        if (d_changed) cudaFree(d_changed);
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        d_row = d_col = d_comp = d_tmp = d_affected = d_upd_u = d_upd_v = d_changed = nullptr;
        ev0 = ev1 = nullptr;
        cap_n = cap_m = cap_upd = 0;
        mem_bytes = 0;
    }
};

#endif

