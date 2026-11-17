/*
 * Group 16 - GPU-Accelerated Dynamic Graph Algorithms
 * Course: CS5658 GPU Programming
 *
 * Team Members:
 * - Arun Shyam (142301007)
 * - D. Lakshmi Harsha (142301008)
 * - K. Aravind (112301017)
 * - B. Lepakshi Kalyan (112301003)
 *
 * This file contains the main driver and benchmark harness.
 * Takes graph and update batch plain text files (.txt) as input, runs both CPU
 * baseline and CUDA kernels, verifies correctness, and reports execution timings.
 */

#include "cpu_algo.h"
#include "triangle.cu"
#include "components.cu"
#include <iomanip>

// Helper to print basic GPU hardware specs
static void print_gpu_info() {
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        printf("GPU Device: %s (Compute %d.%d, %.1f MB free / %.1f MB total)\n",
               prop.name, prop.major, prop.minor,
               free_b / (1024.0 * 1024.0), total_b / (1024.0 * 1024.0));
    }
}

// Structure to store benchmark metrics
struct bench_result {
    string algo;
    string dataset;
    int n = 0;
    long long m = 0;
    int total_upd = 0;
    int batch_sz = 0;
    double cpu_static_ms = 0.0;
    double gpu_static_k_ms = 0.0;
    double gpu_static_e2e_ms = 0.0;
    double cpu_dyn_ms = 0.0;
    double gpu_dyn_k_ms = 0.0;
    double gpu_dyn_e2e_ms = 0.0;
    bool ok = false;
};

static double calc_throughput(int total_upd, double ms) {
    if (total_upd <= 0 || ms <= 0.0) return 0.0;
    return (double)total_upd / (ms / 1000.0);
}

// Run Dynamic Triangle Counting on CPU and GPU
static bench_result run_tc(const string& name,
                           const string& gpath,
                           const string& upath,
                           int batch_sz,
                           bool verbose) {
    if (batch_sz <= 0) batch_sz = 1;
    int skipped = 0;
    dyn_graph g_cpu = load_graph(gpath, &skipped);
    dyn_graph g_gpu = g_cpu;
    csr_graph csr0 = g_gpu.to_csr();

    // 1. CPU static counting
    my_timer tm;
    tm.start();
    long long cpu_tri = cpu_static_tc(g_cpu);
    double cpu_static_ms = tm.stop_ms();

    // 2. GPU static counting
    gpu_tc_state gpu;
    double gpu_static_k_ms = 0.0;
    double gpu_static_e2e_ms = 0.0;
    long long gpu_tri = gpu.static_count(csr0, gpu_static_k_ms, gpu_static_e2e_ms);

    bool ok = (cpu_tri == gpu_tri);

    if (verbose) {
        printf("\n=== Dynamic Triangle Counting (%s) ===\n", name.c_str());
        printf("Graph: n=%d vertices, edges=%lld (skipped invalid=%d)\n",
               g_cpu.n, g_cpu.num_edges, skipped);
        printf("Static Count: CPU=%lld (%.3f ms) | GPU=%lld (kernel=%.3f ms, e2e=%.3f ms)\n",
               cpu_tri, cpu_static_ms, gpu_tri, gpu_static_k_ms, gpu_static_e2e_ms);
        printf("Static Correctness: %s\n", ok ? "PASS" : "FAIL");
    }

    // 3. Dynamic Batch Updates
    vector<edge_op> ops;
    if (!upath.empty()) {
        ops = load_updates(upath);
    }

    double cpu_dyn_ms = 0.0;
    double gpu_dyn_k_ms = 0.0;
    double gpu_dyn_e2e_ms = 0.0;
    long long cpu_cur = cpu_tri;
    long long gpu_cur = gpu_tri;

    if (!ops.empty()) {
        int bid = 0;
        for (size_t s = 0; s < ops.size(); s += (size_t)batch_sz) {
            size_t e = min(s + (size_t)batch_sz, ops.size());
            vector<edge_op> batch(ops.begin() + s, ops.begin() + e);

            // CPU update
            tm.start();
            cpu_dynamic_tc(g_cpu, batch, cpu_cur);
            cpu_dyn_ms += tm.stop_ms();

            // GPU update
            gpu.apply_batch(g_gpu, batch, gpu_cur, gpu_dyn_k_ms, gpu_dyn_e2e_ms);

            bool b_ok = (cpu_cur == gpu_cur);
            ok = ok && b_ok;
            bid++;
            if (verbose && (bid <= 5 || bid % 10 == 0 || s + batch_sz >= ops.size())) {
                printf("  Batch %3d: CPU=%lld  GPU=%lld  [%s]\n",
                       bid, cpu_cur, gpu_cur, b_ok ? "PASS" : "FAIL");
            }
        }

        // Final full recount check
        double rec_k = 0.0, rec_e2e = 0.0;
        long long cpu_rec = cpu_static_tc(g_cpu);
        long long gpu_rec = gpu.static_count(g_gpu.to_csr(), rec_k, rec_e2e);
        bool rec_ok = (cpu_rec == gpu_rec && cpu_rec == cpu_cur && gpu_rec == gpu_cur);
        ok = ok && rec_ok;

        if (verbose) {
            printf("\nDynamic Updates Summary (%d updates, batch size=%d):\n",
                   (int)ops.size(), batch_sz);
            printf("  CPU Total Time: %.3f ms (%.0f upd/s)\n",
                   cpu_dyn_ms, calc_throughput((int)ops.size(), cpu_dyn_ms));
            printf("  GPU Kernel Time: %.3f ms (%.0f upd/s)\n",
                   gpu_dyn_k_ms, calc_throughput((int)ops.size(), gpu_dyn_k_ms));
            printf("  GPU End-to-End:  %.3f ms (%.0f upd/s)\n",
                   gpu_dyn_e2e_ms, calc_throughput((int)ops.size(), gpu_dyn_e2e_ms));
            if (gpu_dyn_k_ms > 0.0) {
                printf("  Kernel Speedup:  %.2fx\n", cpu_dyn_ms / gpu_dyn_k_ms);
            }
            printf("  Final Recount Check: %s\n", rec_ok ? "PASS" : "FAIL");
        }
    }

    bench_result res;
    res.algo = "TC";
    res.dataset = name;
    res.n = g_cpu.n;
    res.m = csr0.m / 2;
    res.total_upd = (int)ops.size();
    res.batch_sz = batch_sz;
    res.cpu_static_ms = cpu_static_ms;
    res.gpu_static_k_ms = gpu_static_k_ms;
    res.gpu_static_e2e_ms = gpu_static_e2e_ms;
    res.cpu_dyn_ms = cpu_dyn_ms;
    res.gpu_dyn_k_ms = gpu_dyn_k_ms;
    res.gpu_dyn_e2e_ms = gpu_dyn_e2e_ms;
    res.ok = ok;

    gpu.free_all();
    return res;
}

// Run Dynamic Connected Components on CPU and GPU
static bench_result run_cc(const string& name,
                           const string& gpath,
                           const string& upath,
                           int batch_sz,
                           bool verbose) {
    if (batch_sz <= 0) batch_sz = 1;
    int skipped = 0;
    dyn_graph g_cpu = load_graph(gpath, &skipped);
    dyn_graph g_gpu = g_cpu;
    csr_graph csr0 = g_gpu.to_csr();

    // 1. CPU static connected components
    vector<int> cpu_comp;
    my_timer tm;
    tm.start();
    int cpu_num_cc = cpu_static_cc(g_cpu, cpu_comp);
    double cpu_static_ms = tm.stop_ms();

    // 2. GPU static connected components
    gpu_cc_state gpu;
    vector<int> gpu_comp;
    double gpu_static_k_ms = 0.0;
    double gpu_static_e2e_ms = 0.0;
    int gpu_num_cc = gpu.static_cc(csr0, gpu_comp, gpu_static_k_ms, gpu_static_e2e_ms);

    bool ok = (cpu_num_cc == gpu_num_cc);

    if (verbose) {
        printf("\n=== Dynamic Connected Components (%s) ===\n", name.c_str());
        printf("Graph: n=%d vertices, edges=%lld\n", g_cpu.n, g_cpu.num_edges);
        printf("Static CC Count: CPU=%d (%.3f ms) | GPU=%d (kernel=%.3f ms, e2e=%.3f ms)\n",
               cpu_num_cc, cpu_static_ms, gpu_num_cc, gpu_static_k_ms, gpu_static_e2e_ms);
        printf("Static Correctness: %s\n", ok ? "PASS" : "FAIL");
    }

    // 3. Dynamic Updates
    vector<edge_op> ops;
    if (!upath.empty()) {
        ops = load_updates(upath);
    }

    double cpu_dyn_ms = 0.0;
    double gpu_dyn_k_ms = 0.0;
    double gpu_dyn_e2e_ms = 0.0;
    int cpu_cur_cc = cpu_num_cc;
    int gpu_cur_cc = gpu_num_cc;

    if (!ops.empty()) {
        int bid = 0;
        for (size_t s = 0; s < ops.size(); s += (size_t)batch_sz) {
            size_t e = min(s + (size_t)batch_sz, ops.size());
            vector<edge_op> batch(ops.begin() + s, ops.begin() + e);

            // CPU update
            tm.start();
            cpu_cur_cc = cpu_dynamic_cc(g_cpu, batch, cpu_comp);
            cpu_dyn_ms += tm.stop_ms();

            // GPU update
            gpu_cur_cc = gpu.apply_batch(g_gpu, batch, gpu_comp, gpu_dyn_k_ms, gpu_dyn_e2e_ms);

            bool b_ok = (cpu_cur_cc == gpu_cur_cc);
            ok = ok && b_ok;
            bid++;
            if (verbose && (bid <= 5 || bid % 10 == 0 || s + batch_sz >= ops.size())) {
                printf("  Batch %3d: CPU CC=%d  GPU CC=%d  [%s]\n",
                       bid, cpu_cur_cc, gpu_cur_cc, b_ok ? "PASS" : "FAIL");
            }
        }

        if (verbose) {
            printf("\nDynamic Updates Summary (%d updates, batch size=%d):\n",
                   (int)ops.size(), batch_sz);
            printf("  CPU Total Time: %.3f ms (%.0f upd/s)\n",
                   cpu_dyn_ms, calc_throughput((int)ops.size(), cpu_dyn_ms));
            printf("  GPU Kernel Time: %.3f ms (%.0f upd/s)\n",
                   gpu_dyn_k_ms, calc_throughput((int)ops.size(), gpu_dyn_k_ms));
            printf("  GPU End-to-End:  %.3f ms (%.0f upd/s)\n",
                   gpu_dyn_e2e_ms, calc_throughput((int)ops.size(), gpu_dyn_e2e_ms));
            if (gpu_dyn_k_ms > 0.0) {
                printf("  Kernel Speedup:  %.2fx\n", cpu_dyn_ms / gpu_dyn_k_ms);
            }
        }
    }

    bench_result res;
    res.algo = "CC";
    res.dataset = name;
    res.n = g_cpu.n;
    res.m = csr0.m / 2;
    res.total_upd = (int)ops.size();
    res.batch_sz = batch_sz;
    res.cpu_static_ms = cpu_static_ms;
    res.gpu_static_k_ms = gpu_static_k_ms;
    res.gpu_static_e2e_ms = gpu_static_e2e_ms;
    res.cpu_dyn_ms = cpu_dyn_ms;
    res.gpu_dyn_k_ms = gpu_dyn_k_ms;
    res.gpu_dyn_e2e_ms = gpu_dyn_e2e_ms;
    res.ok = ok;

    gpu.free_all();
    return res;
}

int main(int argc, char** argv) {
    printf("====================================================\n");
    printf(" Group 16: GPU-Accelerated Dynamic Graph Algorithms \n");
    printf("====================================================\n");
    print_gpu_info();

    string mode = "both";
    string gpath = "";
    string upath = "";
    int batch_sz = 25;

    for (int i = 1; i < argc; ++i) {
        string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            printf("\nUsage:\n");
            printf("  %s [tc|cc|both] --graph <graph.txt> [--updates <updates.txt>] [--batch <size>]\n\n", argv[0]);
            printf("Options:\n");
            printf("  tc        Run Dynamic Triangle Counting\n");
            printf("  cc        Run Dynamic Connected Components\n");
            printf("  both      Run both algorithms (default)\n");
            printf("  --graph   Path to graph text file (.txt)\n");
            printf("  --updates Path to updates text file (.txt)\n");
            printf("  --batch   Batch size for dynamic updates (default: 25)\n");
            return 0;
        } else if (i == 1 && arg[0] != '-') {
            mode = arg;
        } else if (arg == "--graph" && i + 1 < argc) {
            gpath = argv[++i];
        } else if (arg == "--updates" && i + 1 < argc) {
            upath = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_sz = atoi(argv[++i]);
        }
    }

    if (gpath.empty()) {
        printf("\nNote: No --graph provided. Checking default text datasets...\n");
        ifstream test_f("data/small.txt");
        if (test_f.good()) {
            gpath = "data/small.txt";
            upath = "data/small_updates.txt";
        } else {
            ifstream test_el("data/small.el");
            if (test_el.good()) {
                gpath = "data/small.el";
                upath = "data/small.upd";
            }
        }
    }

    if (gpath.empty()) {
        fprintf(stderr, "Error: Please specify graph file using --graph <path.txt>\n");
        printf("Run with -h or --help for instructions.\n");
        return 1;
    }

    bool all_ok = true;

    if (mode == "tc" || mode == "both") {
        bench_result r = run_tc(gpath, gpath, upath, batch_sz, true);
        all_ok = all_ok && r.ok;
    }

    if (mode == "cc" || mode == "both") {
        bench_result r = run_cc(gpath, gpath, upath, batch_sz, true);
        all_ok = all_ok && r.ok;
    }

    printf("\n====================================================\n");
    printf(" Overall Execution: %s\n", all_ok ? "ALL TESTS PASSED" : "FAILED");
    printf("====================================================\n");

    return all_ok ? 0 : 1;
}

