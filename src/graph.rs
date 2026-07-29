//! Directed-graph utilities for plan and dependency analysis.
//!
//! A physical plan is a DAG of operators, and the transaction manager tracks a
//! wait-for graph among transactions. Both want the same primitives: topological
//! ordering (and cycle detection), strongly connected components, and reachability.
//! This module provides a compact adjacency-list [`DiGraph`] over dense integer
//! vertex ids plus those algorithms, all iterative so deep graphs do not blow
//! the stack.

/// A directed graph over `0..n` vertices.
#[derive(Debug, Clone, Default)]
pub struct DiGraph {
    adj: Vec<Vec<usize>>,
}

impl DiGraph {
    /// A graph with `n` isolated vertices.
    pub fn new(n: usize) -> DiGraph {
        DiGraph {
            adj: vec![Vec::new(); n],
        }
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.adj.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.adj.iter().map(|v| v.len()).sum()
    }

    /// Ensure the graph has at least `n` vertices.
    pub fn ensure_vertex(&mut self, v: usize) {
        if self.adj.len() <= v {
            self.adj.resize(v + 1, Vec::new());
        }
    }

    /// Add a directed edge `from -> to`.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.ensure_vertex(from.max(to));
        self.adj[from].push(to);
    }

    /// Successors of a vertex.
    pub fn successors(&self, v: usize) -> &[usize] {
        &self.adj[v]
    }

    /// In-degree of each vertex.
    pub fn in_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.adj.len()];
        for edges in &self.adj {
            for &to in edges {
                deg[to] += 1;
            }
        }
        deg
    }

    /// Kahn's algorithm topological sort. Returns `None` if there is a cycle.
    pub fn topo_sort(&self) -> Option<Vec<usize>> {
        let mut indeg = self.in_degrees();
        let mut queue: Vec<usize> = (0..self.adj.len()).filter(|&v| indeg[v] == 0).collect();
        // Use a stable order for reproducibility.
        queue.sort_unstable();
        let mut order = Vec::with_capacity(self.adj.len());
        let mut head = 0;
        while head < queue.len() {
            let v = queue[head];
            head += 1;
            order.push(v);
            for &to in &self.adj[v] {
                indeg[to] -= 1;
                if indeg[to] == 0 {
                    queue.push(to);
                }
            }
        }
        if order.len() == self.adj.len() {
            Some(order)
        } else {
            None
        }
    }

    /// `true` if the graph contains a directed cycle.
    pub fn has_cycle(&self) -> bool {
        self.topo_sort().is_none()
    }

    /// `true` if `dst` is reachable from `src` via directed edges.
    pub fn reachable(&self, src: usize, dst: usize) -> bool {
        if src >= self.adj.len() {
            return false;
        }
        let mut visited = vec![false; self.adj.len()];
        let mut stack = vec![src];
        visited[src] = true;
        while let Some(v) = stack.pop() {
            if v == dst {
                return true;
            }
            for &to in &self.adj[v] {
                if !visited[to] {
                    visited[to] = true;
                    stack.push(to);
                }
            }
        }
        false
    }

    /// Tarjan's strongly connected components, each returned as a sorted vertex
    /// list; components are in reverse topological order.
    pub fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.adj.len();
        let mut index = vec![usize::MAX; n];
        let mut lowlink = vec![0usize; n];
        let mut on_stack = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut counter = 0usize;

        // Iterative Tarjan using an explicit work stack of (vertex, edge index).
        for start in 0..n {
            if index[start] != usize::MAX {
                continue;
            }
            let mut work: Vec<(usize, usize)> = vec![(start, 0)];
            while let Some(&(v, ei)) = work.last() {
                if ei == 0 {
                    index[v] = counter;
                    lowlink[v] = counter;
                    counter += 1;
                    stack.push(v);
                    on_stack[v] = true;
                }
                if ei < self.adj[v].len() {
                    let w = self.adj[v][ei];
                    work.last_mut().unwrap().1 += 1;
                    if index[w] == usize::MAX {
                        work.push((w, 0));
                    } else if on_stack[w] {
                        lowlink[v] = lowlink[v].min(index[w]);
                    }
                } else {
                    // Done with v; propagate lowlink to parent and maybe close SCC.
                    if lowlink[v] == index[v] {
                        let mut comp = Vec::new();
                        loop {
                            let w = stack.pop().unwrap();
                            on_stack[w] = false;
                            comp.push(w);
                            if w == v {
                                break;
                            }
                        }
                        comp.sort_unstable();
                        components.push(comp);
                    }
                    work.pop();
                    if let Some(&(parent, _)) = work.last() {
                        lowlink[parent] = lowlink[parent].min(lowlink[v]);
                    }
                }
            }
        }
        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topo_of_dag() {
        let mut g = DiGraph::new(6);
        g.add_edge(5, 2);
        g.add_edge(5, 0);
        g.add_edge(4, 0);
        g.add_edge(4, 1);
        g.add_edge(2, 3);
        g.add_edge(3, 1);
        let order = g.topo_sort().unwrap();
        // Verify it's a valid ordering.
        let pos: Vec<usize> = {
            let mut p = vec![0; 6];
            for (i, &v) in order.iter().enumerate() {
                p[v] = i;
            }
            p
        };
        for from in 0..6 {
            for &to in g.successors(from) {
                assert!(pos[from] < pos[to]);
            }
        }
    }

    #[test]
    fn detects_cycle() {
        let mut g = DiGraph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        assert!(g.has_cycle());
        assert!(g.topo_sort().is_none());
    }

    #[test]
    fn reachability() {
        let mut g = DiGraph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        assert!(g.reachable(0, 2));
        assert!(!g.reachable(2, 0));
        assert!(!g.reachable(0, 3));
    }

    #[test]
    fn scc_components() {
        let mut g = DiGraph::new(5);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0); // cycle {0,1,2}
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        let mut comps = g.strongly_connected_components();
        comps.sort_by_key(|c| c[0]);
        assert!(comps.contains(&vec![0, 1, 2]));
        assert!(comps.contains(&vec![3]));
        assert!(comps.contains(&vec![4]));
        assert_eq!(comps.len(), 3);
    }

    #[test]
    fn dynamic_growth() {
        let mut g = DiGraph::new(0);
        g.add_edge(2, 5);
        assert_eq!(g.vertex_count(), 6);
        assert_eq!(g.edge_count(), 1);
    }
}
