#ifndef CPU_ALGO_H
#define CPU_ALGO_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstdio>
#include <cstdlib>

using namespace std;

// timer helper
struct my_timer {
    chrono::high_resolution_clock::time_point t0;

    void start() {
        t0 = chrono::high_resolution_clock::now();
    }

    double stop_ms() const {
        auto t1 = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(t1 - t0).count();
    }
};

// edge update: I = insert, D = delete
struct edge_op {
    int u;
    int v;
    bool ins;
};

struct edge_pair {
    int u;
    int v;

    bool operator<(const edge_pair& other) const {
        if (u != other.u) return u < other.u;
        return v < other.v;
    }

    bool operator==(const edge_pair& other) const {
        return u == other.u && v == other.v;
    }
};

// undirected graph in csr format
struct csr_graph {
    int n = 0;
    int m = 0;
    vector<int> row;
    vector<int> col;
    vector<int> deg;

    void from_adj(const vector<vector<int>>& adj) {
        n = (int)adj.size();
        row.assign(n + 1, 0);
        deg.assign(n, 0);
        for (int u = 0; u < n; ++u) {
            deg[u] = (int)adj[u].size();
            row[u + 1] = row[u] + deg[u];
        }
        m = row[n];
        col.resize(m);
        for (int u = 0; u < n; ++u) {
            for (int k = 0; k < deg[u]; ++k) {
                col[row[u] + k] = adj[u][k];
            }
        }
    }

    size_t bytes() const {
        return row.size() * sizeof(int) + col.size() * sizeof(int);
    }
};

// dynamic graph with sorted adjacency list
struct dyn_graph {
    int n = 0;
    long long num_edges = 0;
    vector<vector<int>> adj;

    dyn_graph() = default;
    explicit dyn_graph(int num_v) : n(num_v), num_edges(0), adj(num_v) {}

    bool valid_edge(int u, int v) const {
        if (u < 0 || u >= n || v < 0 || v >= n) return false;
        if (u == v) return false;
        return true;
    }

    bool has_edge(int u, int v) const {
        if (!valid_edge(u, v)) return false;
        if (adj[u].size() > adj[v].size()) {
            return binary_search(adj[v].begin(), adj[v].end(), u);
        }
        return binary_search(adj[u].begin(), adj[u].end(), v);
    }

    bool add_edge(int u, int v) {
        if (!valid_edge(u, v)) return false;
        if (has_edge(u, v)) return false;

        auto it_u = lower_bound(adj[u].begin(), adj[u].end(), v);
        adj[u].insert(it_u, v);

        auto it_v = lower_bound(adj[v].begin(), adj[v].end(), u);
        adj[v].insert(it_v, u);

        num_edges++;
        return true;
    }

    bool del_edge(int u, int v) {
        if (!valid_edge(u, v)) return false;
        auto it_u = lower_bound(adj[u].begin(), adj[u].end(), v);
        if (it_u == adj[u].end() || *it_u != v) return false;
        adj[u].erase(it_u);

        auto it_v = lower_bound(adj[v].begin(), adj[v].end(), u);
        if (it_v != adj[v].end() && *it_v == u) {
            adj[v].erase(it_v);
        }

        num_edges--;
        return true;
    }

    csr_graph to_csr() const {
        csr_graph csr;
        csr.from_adj(adj);
        return csr;
    }
};

// degree check to avoid double counting
inline bool deg_less(int a, int deg_a, int b, int deg_b) {
    return (deg_a < deg_b) || (deg_a == deg_b && a < b);
}

// count common neighbors between two sorted lists
inline long long intersect_cnt(const vector<int>& a, const vector<int>& b) {
    if (a.empty() || b.empty()) return 0;

    if (a.size() * 8 < b.size()) {
        long long cnt = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (binary_search(b.begin(), b.end(), a[i])) cnt++;
        }
        return cnt;
    }
    if (b.size() * 8 < a.size()) {
        long long cnt = 0;
        for (size_t i = 0; i < b.size(); ++i) {
            if (binary_search(a.begin(), a.end(), b[i])) cnt++;
        }
        return cnt;
    }

    size_t ia = 0, ib = 0;
    long long cnt = 0;
    while (ia < a.size() && ib < b.size()) {
        if (a[ia] < b[ib]) {
            ia++;
        } else if (a[ia] > b[ib]) {
            ib++;
        } else {
            cnt++;
            ia++;
            ib++;
        }
    }
    return cnt;
}

// static triangle counting on cpu
inline long long cpu_static_tc(const dyn_graph& g) {
    long long total = 0;
    for (int u = 0; u < g.n; ++u) {
        int deg_u = (int)g.adj[u].size();
        for (size_t i = 0; i < g.adj[u].size(); ++i) {
            int v = g.adj[u][i];
            int deg_v = (int)g.adj[v].size();
            if (!deg_less(u, deg_u, v, deg_v)) continue;

            for (size_t j = 0; j < g.adj[u].size(); ++j) {
                int w = g.adj[u][j];
                int deg_w = (int)g.adj[w].size();
                if (!deg_less(v, deg_v, w, deg_w)) continue;
                if (binary_search(g.adj[v].begin(), g.adj[v].end(), w)) {
                    total++;
                }
            }
        }
    }
    return total;
}

// dynamic triangle counting updates
inline void cpu_dynamic_tc(dyn_graph& g,
                           const vector<edge_op>& batch,
                           long long& tri_cnt) {
    for (size_t i = 0; i < batch.size(); ++i) {
        const edge_op& e = batch[i];
        if (!g.valid_edge(e.u, e.v)) continue;
        int u = min(e.u, e.v);
        int v = max(e.u, e.v);

        if (e.ins) {
            if (g.has_edge(u, v)) continue;
            long long delta = intersect_cnt(g.adj[u], g.adj[v]);
            g.add_edge(u, v);
            tri_cnt += delta;
        } else {
            if (!g.has_edge(u, v)) continue;
            long long delta = intersect_cnt(g.adj[u], g.adj[v]);
            g.del_edge(u, v);
            tri_cnt -= delta;
        }
    }
}

// union find with path compression
struct union_find {
    vector<int> parent;
    vector<int> rnk;
    int num_cc = 0;

    void init(int n) {
        parent.resize(n);
        rnk.assign(n, 0);
        num_cc = n;
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int x) {
        int root = x;
        while (root != parent[root]) root = parent[root];
        int curr = x;
        while (curr != root) {
            int nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    }

    bool unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return false;

        if (rnk[ra] < rnk[rb]) {
            parent[ra] = rb;
        } else if (rnk[ra] > rnk[rb]) {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rnk[ra]++;
        }
        num_cc--;
        return true;
    }
};

// static connected components on cpu
inline int cpu_static_cc(const dyn_graph& g, vector<int>& comp) {
    union_find uf;
    uf.init(g.n);
    for (int u = 0; u < g.n; ++u) {
        for (size_t i = 0; i < g.adj[u].size(); ++i) {
            int v = g.adj[u][i];
            if (u < v) uf.unite(u, v);
        }
    }
    comp.resize(g.n);
    for (int i = 0; i < g.n; ++i) comp[i] = uf.find(i);
    return uf.num_cc;
}

// dynamic connected components updates
inline int cpu_dynamic_cc(dyn_graph& g,
                          const vector<edge_op>& batch,
                          vector<int>& comp) {
    bool has_del = false;
    for (size_t i = 0; i < batch.size(); ++i) {
        if (!batch[i].ins && g.has_edge(batch[i].u, batch[i].v)) {
            has_del = true;
            break;
        }
    }

    if (has_del) {
        for (size_t i = 0; i < batch.size(); ++i) {
            if (batch[i].ins) g.add_edge(batch[i].u, batch[i].v);
            else g.del_edge(batch[i].u, batch[i].v);
        }
        return cpu_static_cc(g, comp);
    }

    union_find uf;
    uf.init(g.n);
    for (int i = 0; i < g.n; ++i) uf.parent[i] = comp[i];

    for (size_t i = 0; i < batch.size(); ++i) {
        const edge_op& e = batch[i];
        if (e.ins && g.valid_edge(e.u, e.v)) {
            if (g.add_edge(e.u, e.v)) {
                uf.unite(e.u, e.v);
            }
        }
    }

    for (int i = 0; i < g.n; ++i) comp[i] = uf.find(i);

    vector<int> roots;
    roots.reserve(g.n);
    for (int i = 0; i < g.n; ++i) roots.push_back(comp[i]);
    sort(roots.begin(), roots.end());
    roots.erase(unique(roots.begin(), roots.end()), roots.end());
    return (int)roots.size();
}

// read graph from text file
inline dyn_graph load_graph(const string& path, int* skipped = nullptr) {
    ifstream fin(path);
    if (!fin) {
        fprintf(stderr, "cannot open graph file %s\n", path.c_str());
        exit(1);
    }

    int n = 0;
    long long m = 0;
    if (!(fin >> n >> m) || n < 0 || m < 0) {
        fprintf(stderr, "invalid format in %s\n", path.c_str());
        exit(1);
    }

    dyn_graph g(n);
    int bad_cnt = 0;
    for (long long i = 0; i < m; ++i) {
        int u = 0, v = 0;
        if (!(fin >> u >> v)) break;
        if (!g.add_edge(u, v)) {
            bad_cnt++;
        }
    }
    if (skipped) *skipped = bad_cnt;
    return g;
}

// read dynamic updates from text file
inline vector<edge_op> load_updates(const string& path) {
    ifstream fin(path);
    if (!fin) {
        fprintf(stderr, "cannot open updates file %s\n", path.c_str());
        exit(1);
    }

    vector<edge_op> ops;
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        char op_ch = 0;
        int u = 0, v = 0;
        if (!(iss >> op_ch >> u >> v)) continue;
        if (op_ch == 'I' || op_ch == 'i') {
            ops.push_back({u, v, true});
        } else if (op_ch == 'D' || op_ch == 'd') {
            ops.push_back({u, v, false});
        }
    }
    return ops;
}

#endif
