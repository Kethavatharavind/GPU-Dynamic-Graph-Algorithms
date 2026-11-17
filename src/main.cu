#include "cpu_algo.h"
#include "triangle.cu"
#include "components.cu"
#include <iomanip>

// print gpu specs
static void print_gpu_info() {
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        printf("gpu: %s (compute %d.%d, %.1f mb free)\n",
               prop.name, prop.major, prop.minor,
               free_b / (1024.0 * 1024.0));
    }
}

// benchmark metrics
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

// run dynamic triangle counting
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

    // cpu static
    my_timer tm;
    tm.start();
    long long cpu_tri = cpu_static_tc(g_cpu);
    double cpu_static_ms = tm.stop_ms();

    // gpu static
    gpu_tc_state gpu;
    double gpu_static_k_ms = 0.0;
    double gpu_static_e2e_ms = 0.0;
    long long gpu_tri = gpu.static_count(csr0, gpu_static_k_ms, gpu_static_e2e_ms);

    bool ok = (cpu_tri == gpu_tri);

    if (verbose) {
        printf("\n--- dynamic triangle counting (%s) ---\n", name.c_str());
        printf("graph: n=%d, edges=%lld (skipped=%d)\n", g_cpu.n, g_cpu.num_edges, skipped);
        printf("static cpu: %lld (%.3f ms)\n", cpu_tri, cpu_static_ms);
        printf("static gpu: %lld (kernel %.3f ms, e2e %.3f ms)\n", gpu_tri, gpu_static_k_ms, gpu_static_e2e_ms);
        printf("static check: %s\n", ok ? "pass" : "fail");
    }

    // dynamic updates
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

            tm.start();
            cpu_dynamic_tc(g_cpu, batch, cpu_cur);
            cpu_dyn_ms += tm.stop_ms();

            gpu.apply_batch(g_gpu, batch, gpu_cur, gpu_dyn_k_ms, gpu_dyn_e2e_ms);

            bool b_ok = (cpu_cur == gpu_cur);
            ok = ok && b_ok;
            bid++;
            if (verbose && (bid <= 5 || bid % 10 == 0 || s + batch_sz >= ops.size())) {
                printf("  batch %3d: cpu=%lld gpu=%lld [%s]\n", bid, cpu_cur, gpu_cur, b_ok ? "ok" : "fail");
            }
        }

        // final check
        double rec_k = 0.0, rec_e2e = 0.0;
        long long cpu_rec = cpu_static_tc(g_cpu);
        long long gpu_rec = gpu.static_count(g_gpu.to_csr(), rec_k, rec_e2e);
        bool rec_ok = (cpu_rec == gpu_rec && cpu_rec == cpu_cur && gpu_rec == gpu_cur);
        ok = ok && rec_ok;

        if (verbose) {
            printf("dynamic cpu time: %.3f ms (%.0f upd/s)\n", cpu_dyn_ms, calc_throughput((int)ops.size(), cpu_dyn_ms));
            printf("dynamic gpu time: kernel %.3f ms, e2e %.3f ms\n", gpu_dyn_k_ms, gpu_dyn_e2e_ms);
            if (gpu_dyn_k_ms > 0.0) {
                printf("speedup: %.2fx\n", cpu_dyn_ms / gpu_dyn_k_ms);
            }
            printf("final recount: %s\n", rec_ok ? "pass" : "fail");
        }
    }

    bench_result res;
    res.algo = "tc";
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

// run dynamic connected components
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

    // cpu static cc
    vector<int> cpu_comp;
    my_timer tm;
    tm.start();
    int cpu_num_cc = cpu_static_cc(g_cpu, cpu_comp);
    double cpu_static_ms = tm.stop_ms();

    // gpu static cc
    gpu_cc_state gpu;
    vector<int> gpu_comp;
    double gpu_static_k_ms = 0.0;
    double gpu_static_e2e_ms = 0.0;
    int gpu_num_cc = gpu.static_cc(csr0, gpu_comp, gpu_static_k_ms, gpu_static_e2e_ms);

    bool ok = (cpu_num_cc == gpu_num_cc);

    if (verbose) {
        printf("\n--- dynamic connected components (%s) ---\n", name.c_str());
        printf("graph: n=%d, edges=%lld\n", g_cpu.n, g_cpu.num_edges);
        printf("static cpu cc: %d (%.3f ms)\n", cpu_num_cc, cpu_static_ms);
        printf("static gpu cc: %d (kernel %.3f ms, e2e %.3f ms)\n", gpu_num_cc, gpu_static_k_ms, gpu_static_e2e_ms);
        printf("static check: %s\n", ok ? "pass" : "fail");
    }

    // dynamic updates
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

            tm.start();
            cpu_cur_cc = cpu_dynamic_cc(g_cpu, batch, cpu_comp);
            cpu_dyn_ms += tm.stop_ms();

            gpu_cur_cc = gpu.apply_batch(g_gpu, batch, gpu_comp, gpu_dyn_k_ms, gpu_dyn_e2e_ms);

            bool b_ok = (cpu_cur_cc == gpu_cur_cc);
            ok = ok && b_ok;
            bid++;
            if (verbose && (bid <= 5 || bid % 10 == 0 || s + batch_sz >= ops.size())) {
                printf("  batch %3d: cpu cc=%d gpu cc=%d [%s]\n", bid, cpu_cur_cc, gpu_cur_cc, b_ok ? "ok" : "fail");
            }
        }

        if (verbose) {
            printf("dynamic cpu time: %.3f ms (%.0f upd/s)\n", cpu_dyn_ms, calc_throughput((int)ops.size(), cpu_dyn_ms));
            printf("dynamic gpu time: kernel %.3f ms, e2e %.3f ms\n", gpu_dyn_k_ms, gpu_dyn_e2e_ms);
            if (gpu_dyn_k_ms > 0.0) {
                printf("speedup: %.2fx\n", cpu_dyn_ms / gpu_dyn_k_ms);
            }
        }
    }

    bench_result res;
    res.algo = "cc";
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
    printf("group 16: gpu dynamic graph algorithms\n");
    print_gpu_info();

    string mode = "both";
    string gpath = "";
    string upath = "";
    int batch_sz = 25;

    for (int i = 1; i < argc; ++i) {
        string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            printf("usage: %s [tc|cc|both] --graph <graph.txt> [--updates <updates.txt>] [--batch <size>]\n", argv[0]);
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
        fprintf(stderr, "please specify graph file with --graph <path.txt>\n");
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

    printf("\nall tests: %s\n", all_ok ? "pass" : "fail");
    return all_ok ? 0 : 1;
}
