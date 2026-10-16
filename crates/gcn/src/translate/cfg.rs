//! Structured control flow for branchy programs.
//!
//! GCN code from the title's compiler is structured at the source level
//! (LLVM-style structurization before ISel: single-entry/single-exit `if`
//! regions, natural loops), so most programs map onto SPIR-V selections and
//! loops directly. `plan` builds the CFG, checks that it is reducible, finds
//! natural loops (one exit block each) and lays the program out as a tree of
//! nodes: straight-line blocks, `If` regions merging at their immediate
//! post-dominator (computed per loop region, inner loops collapsed, with
//! continue/break/return as virtual sinks), loops, break, continue, return.
//! A block reached from a second region (typically a shared exit sequence
//! after an early-out branch) is placed again there - its code duplicated -
//! while the duplicated instructions stay within half the program (plus 256):
//! the dispatcher's loop + switch made the largest pixel shaders take 30-47 s
//! to compile. Anything else (multi-exit loops, multi-level breaks,
//! irreducible flow, a duplication over budget) makes `plan` fail and the
//! caller keeps the dispatcher, which runs any CFG. `emit` writes the tree as SPIR-V structured control flow; each
//! construct gets fresh merge/continue blocks, so no two constructs share one.

use super::{R, T, e, translate_inst};
use crate::Inst;
use rspirv::spirv::{self, Word};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Term {
    /// Falls through (or jumps unconditionally) to a block.
    Goto(usize),
    /// `s_cbranch_*` (SOPP op 4..=9): to `taken` when the condition holds.
    Cond { op: u16, taken: usize, fall: usize },
    /// `s_endpgm` or the end of the code.
    End,
}

#[derive(Debug)]
pub(crate) enum Node {
    /// Block `b`'s instructions up to (not including) its terminator.
    Block(usize),
    /// Condition of `op` (evaluated after the preceding `Block`): `then_` runs
    /// when the branch is taken. `merged`: control reaches the code after it.
    If { op: u16, then_: Vec<Node>, else_: Vec<Node>, merged: bool },
    /// A loop; `exits`: whether any path leaves it (to the code after it).
    Loop { body: Vec<Node>, exits: bool },
    Break,
    Continue,
    Return,
}

struct Loop {
    header: usize,
    body: Vec<bool>,
    exit: Option<usize>,
    parent: Option<usize>,
}

/// Region-graph vertex: a block (or a collapsed inner loop by its header) or a sink.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum V {
    B(usize),
    Cont,
    Brk,
    End,
    Exit,
}

pub(crate) struct Plan {
    pub nodes: Vec<Node>,
    pub blocks: usize,
    pub loops: usize,
    pub ifs: usize,
}

struct Planner<'a> {
    term: &'a [Term],
    loops: Vec<Loop>,
    /// Loop index per header block.
    loop_of_header: Vec<Option<usize>>,
    /// Innermost loop containing each block.
    innermost: Vec<Option<usize>>,
    placed: Vec<bool>,
    ifs: usize,
    /// Instructions per block; instructions placed a second time so far,
    /// and the most allowed.
    size: Vec<usize>,
    dup: usize,
    dup_max: usize,
}

fn succs(t: Term) -> Vec<usize> {
    match t {
        Term::Goto(n) => vec![n],
        Term::Cond { taken, fall, .. } => vec![taken, fall],
        Term::End => vec![],
    }
}

/// Builds the CFG over `ranges` (block instruction ranges in `starts` order).
fn build(code: &[Inst], starts: &BTreeSet<u32>, ranges: &[std::ops::Range<usize>]) -> Result<Vec<Term>, String> {
    let index: std::collections::HashMap<u32, usize> = starts.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let block_at = |pc: u32| index.get(&pc).copied().ok_or_else(|| format!("branch to {pc:#x}, not a block start"));
    let mut term = Vec::with_capacity(ranges.len());
    for r in ranges {
        let mut t = match code.get(r.end) {
            Some(n) => Term::Goto(block_at(n.pc)?),
            None => Term::End,
        };
        for k in r.clone() {
            let ins = &code[k];
            if ins.is_endpgm() {
                t = Term::End;
                break;
            }
            if let Some(target) = ins.branch_target() {
                let tb = block_at(target)?;
                t = match ins.op {
                    2 => Term::Goto(tb),
                    op => {
                        let fall = code.get(k + 1).ok_or("conditional branch at the end of the code")?;
                        Term::Cond { op, taken: tb, fall: block_at(fall.pc)? }
                    }
                };
                break;
            }
        }
        term.push(t);
    }
    Ok(term)
}

/// Reverse postorder of the blocks reachable from block 0, and whether a
/// retreating edge exists that is not a back edge (checked after dominators).
fn rpo(term: &[Term]) -> Vec<usize> {
    let n = term.len();
    let mut seen = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    seen[0] = true;
    while let Some(&(b, i)) = stack.last() {
        let s = succs(term[b]);
        if i < s.len() {
            stack.last_mut().unwrap().1 = i + 1;
            let next = s[i];
            if !seen[next] {
                seen[next] = true;
                stack.push((next, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    post.reverse();
    post
}

/// Immediate dominators (Cooper, Harvey, Kennedy) over `order` (RPO).
fn idoms(term: &[Term], order: &[usize]) -> Vec<Option<usize>> {
    let n = term.len();
    let mut rank = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        rank[b] = i;
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &b in order {
        for s in succs(term[b]) {
            preds[s].push(b);
        }
    }
    let mut idom: Vec<Option<usize>> = vec![None; n];
    idom[order[0]] = Some(order[0]);
    let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| {
        while a != b {
            while rank[a] > rank[b] {
                a = idom[a].unwrap();
            }
            while rank[b] > rank[a] {
                b = idom[b].unwrap();
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order[1..] {
            let mut new: Option<usize> = None;
            for &p in &preds[b] {
                if idom[p].is_some() {
                    new = Some(match new {
                        None => p,
                        Some(x) => intersect(&idom, p, x),
                    });
                }
            }
            if new.is_some() && idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}

fn dominates(idom: &[Option<usize>], a: usize, mut b: usize) -> bool {
    loop {
        if a == b {
            return true;
        }
        match idom[b] {
            Some(p) if p != b => b = p,
            _ => return false,
        }
    }
}

pub(crate) fn plan(code: &[Inst], starts: &BTreeSet<u32>, ranges: &[std::ops::Range<usize>]) -> Result<Plan, String> {
    let term = build(code, starts, ranges)?;
    let n = term.len();
    let order = rpo(&term);
    let idom = idoms(&term, &order);
    let reachable: Vec<bool> = (0..n).map(|b| idom[b].is_some()).collect();
    // Back edges (to a dominator) define natural loops; any other edge to a
    // block earlier in RPO is a retreating edge of an irreducible CFG.
    let mut rank = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        rank[b] = i;
    }
    let mut back: Vec<Vec<usize>> = vec![Vec::new(); n]; // header -> latches
    for &b in &order {
        for s in succs(term[b]) {
            if rank[s] <= rank[b] {
                if !dominates(&idom, s, b) {
                    return Err(format!("irreducible control flow (block {b} -> {s})"));
                }
                back[s].push(b);
            }
        }
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &b in &order {
        for s in succs(term[b]) {
            preds[s].push(b);
        }
    }
    // Natural loops, outermost first (headers in RPO).
    let mut loops: Vec<Loop> = Vec::new();
    let mut loop_of_header = vec![None; n];
    for &h in &order {
        if back[h].is_empty() {
            continue;
        }
        let mut body = vec![false; n];
        body[h] = true;
        let mut work: Vec<usize> = back[h].clone();
        while let Some(x) = work.pop() {
            if !body[x] {
                body[x] = true;
                work.extend(preds[x].iter().copied());
            }
        }
        let mut exits: Vec<usize> = Vec::new();
        for b in (0..n).filter(|&b| body[b]) {
            for s in succs(term[b]) {
                if !body[s] && !exits.contains(&s) {
                    exits.push(s);
                }
            }
        }
        if exits.len() > 1 {
            return Err(format!("loop at block {h} has {} exit blocks", exits.len()));
        }
        loop_of_header[h] = Some(loops.len());
        loops.push(Loop { header: h, body, exit: exits.first().copied(), parent: None });
    }
    // Nesting: loops are disjoint or nested (reducible CFG); the parent of a
    // loop is the smallest other loop containing its header.
    let size = |l: &Loop| l.body.iter().filter(|&&x| x).count();
    for i in 0..loops.len() {
        let h = loops[i].header;
        let mut best: Option<(usize, usize)> = None;
        for j in 0..loops.len() {
            if j != i && loops[j].body[h] {
                let s = size(&loops[j]);
                if best.is_none_or(|(_, bs)| s < bs) {
                    best = Some((j, s));
                }
            }
        }
        loops[i].parent = best.map(|(j, _)| j);
    }
    let mut innermost = vec![None; n];
    for b in 0..n {
        let mut best: Option<(usize, usize)> = None;
        for (j, l) in loops.iter().enumerate() {
            if l.body[b] {
                let s = size(l);
                if best.is_none_or(|(_, bs)| s < bs) {
                    best = Some((j, s));
                }
            }
        }
        innermost[b] = best.map(|(j, _)| j);
    }
    let _ = reachable;
    let size: Vec<usize> = ranges.iter().map(|r| r.len()).collect();
    let dup_max = 256 + size.iter().sum::<usize>() / 2;
    let mut p = Planner { term: &term, loops, loop_of_header, innermost, placed: vec![false; n], ifs: 0, size, dup: 0, dup_max };
    let mut nodes = Vec::new();
    p.block(0, None, None, &mut nodes)?;
    let blocks = p.placed.iter().filter(|&&x| x).count();
    Ok(Plan { nodes, blocks, loops: p.loops.len(), ifs: p.ifs })
}

impl Planner<'_> {
    /// Where an edge to `t` goes from inside context `ctx` (a loop or the top level).
    fn map(&self, t: usize, ctx: Option<usize>) -> Result<V, String> {
        if let Some(l) = ctx {
            if t == self.loops[l].header {
                return Ok(V::Cont);
            }
            if Some(t) == self.loops[l].exit {
                return Ok(V::Brk);
            }
        }
        if self.innermost[t] == ctx {
            return Ok(V::B(t));
        }
        if let Some(l) = self.loop_of_header[t] {
            if self.loops[l].parent == ctx {
                return Ok(V::B(t));
            }
        }
        Err(format!("edge to block {t} leaves its region (multi-level exit or entry into a loop body)"))
    }

    /// Successors of region vertex `v` in context `ctx` (inner loops collapsed).
    fn region_succs(&self, b: usize, ctx: Option<usize>) -> Result<Vec<V>, String> {
        if let Some(l) = self.loop_of_header[b] {
            if Some(l) != ctx {
                return Ok(match self.loops[l].exit {
                    Some(x) => vec![self.map(x, ctx)?],
                    None => vec![V::End],
                });
            }
        }
        match self.term[b] {
            Term::End => Ok(vec![V::End]),
            t => succs(t).into_iter().map(|s| self.map(s, ctx)).collect(),
        }
    }

    /// Immediate post-dominator of block `b` in its region (`ctx`): the block
    /// both arms of its branch rejoin at, or None when they only meet at a
    /// continue / break / return.
    fn merge_of(&self, b: usize, ctx: Option<usize>) -> Result<Option<usize>, String> {
        // Vertices reachable from b in the region graph (a DAG), in postorder.
        let mut post: Vec<V> = Vec::new();
        let mut seen: std::collections::HashSet<V> = std::collections::HashSet::new();
        let mut succ_cache: std::collections::HashMap<V, Vec<V>> = std::collections::HashMap::new();
        let mut stack: Vec<(V, usize)> = vec![(V::B(b), 0)];
        seen.insert(V::B(b));
        while let Some(&(v, i)) = stack.last() {
            let ss = match succ_cache.get(&v) {
                Some(s) => s.clone(),
                None => {
                    let s = match v {
                        V::B(x) => self.region_succs(x, ctx)?,
                        V::Cont | V::Brk | V::End => vec![V::Exit],
                        V::Exit => vec![],
                    };
                    succ_cache.insert(v, s.clone());
                    s
                }
            };
            if i < ss.len() {
                stack.last_mut().unwrap().1 += 1;
                let w = ss[i];
                if seen.insert(w) {
                    stack.push((w, 0));
                } else if stack.iter().any(|&(x, _)| x == w) {
                    return Err(format!("cycle in the region of block {b}"));
                }
            } else {
                post.push(v);
                stack.pop();
            }
        }
        // Post-dominator tree over the DAG: children before parents in `post`.
        let mut ipdom: std::collections::HashMap<V, V> = std::collections::HashMap::new();
        let mut depth: std::collections::HashMap<V, usize> = std::collections::HashMap::new();
        ipdom.insert(V::Exit, V::Exit);
        depth.insert(V::Exit, 0);
        for &v in &post {
            if v == V::Exit {
                continue;
            }
            let ss = &succ_cache[&v];
            let mut cur = ss[0];
            for &s in &ss[1..] {
                let (mut a, mut c) = (cur, s);
                while a != c {
                    if depth[&a] >= depth[&c] {
                        a = ipdom[&a];
                    } else {
                        c = ipdom[&c];
                    }
                }
                cur = a;
            }
            ipdom.insert(v, cur);
            depth.insert(v, depth[&cur] + 1);
        }
        Ok(match ipdom[&V::B(b)] {
            V::B(m) => Some(m),
            _ => None,
        })
    }

    /// Follows an edge to `t` within `ctx`, appending to `out`; `stop` ends
    /// the sequence (the enclosing construct's merge). Returns whether
    /// control continues after `out` (false once it ends at a jump).
    fn edge(&mut self, t: usize, stop: Option<usize>, ctx: Option<usize>, out: &mut Vec<Node>) -> Result<(), String> {
        if Some(t) == stop {
            return Ok(());
        }
        match self.map(t, ctx)? {
            V::Cont => out.push(Node::Continue),
            V::Brk => out.push(Node::Break),
            V::B(b) => self.block(b, stop, ctx, out)?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn block(&mut self, b: usize, stop: Option<usize>, ctx: Option<usize>, out: &mut Vec<Node>) -> Result<(), String> {
        if let Some(l) = self.loop_of_header[b] {
            if Some(l) != ctx {
                // Entering the loop at its header.
                let mut body = Vec::new();
                self.block_in(b, None, Some(l), &mut body)?;
                let exit = self.loops[l].exit;
                out.push(Node::Loop { body, exits: exit.is_some() });
                return match exit {
                    Some(x) => self.edge(x, stop, ctx, out),
                    None => Ok(()),
                };
            }
        }
        self.block_in(b, stop, ctx, out)
    }

    fn block_in(&mut self, b: usize, stop: Option<usize>, ctx: Option<usize>, out: &mut Vec<Node>) -> Result<(), String> {
        if self.placed[b] {
            // Reached from a second region: its code again here.
            self.dup += self.size[b];
            if self.dup > self.dup_max {
                return Err(format!("block {b} is reached from two regions (duplicating more than {} instructions)", self.dup_max));
            }
        }
        self.placed[b] = true;
        out.push(Node::Block(b));
        match self.term[b] {
            Term::End => {
                out.push(Node::Return);
                Ok(())
            }
            Term::Goto(n) => self.edge(n, stop, ctx, out),
            Term::Cond { op, taken, fall } => {
                self.ifs += 1;
                let m = self.merge_of(b, ctx)?;
                let arm_stop = m.or(stop);
                let (mut then_, mut else_) = (Vec::new(), Vec::new());
                self.edge(taken, arm_stop, ctx, &mut then_)?;
                self.edge(fall, arm_stop, ctx, &mut else_)?;
                let merged = m.is_some() || [&then_, &else_].iter().any(|a| !ends_in_jump(a));
                out.push(Node::If { op, then_, else_, merged });
                match m {
                    Some(mb) => self.edge(mb, stop, ctx, out),
                    None => Ok(()),
                }
            }
        }
    }
}

/// Whether a node sequence ends in a jump on every path (nothing follows it).
fn ends_in_jump(nodes: &[Node]) -> bool {
    match nodes.last() {
        Some(Node::Break | Node::Continue | Node::Return) => true,
        Some(Node::If { merged, .. }) => !merged,
        Some(Node::Loop { exits, .. }) => !exits,
        _ => false,
    }
}

/// The branch condition of `s_cbranch_*` op (taken when true).
pub(crate) fn branch_cond(tr: &mut T, op: u16) -> R<Word> {
    Ok(match op {
        4 => {
            let s = tr.get_scc()?;
            e(tr.b.logical_not(tr.t.bool_, None, s))?
        }
        5 => tr.get_scc()?,
        6..=9 => {
            let src = if op <= 7 { crate::OP_VCC_LO } else { crate::OP_EXEC_LO };
            let lo = tr.sreg(src as u32)?;
            let hi = tr.sreg(src as u32 + 1)?;
            let o = e(tr.b.bitwise_or(tr.t.u32, None, lo, hi))?;
            let z = tr.u(0);
            let zero = e(tr.b.i_equal(tr.t.bool_, None, o, z))?;
            if op == 6 || op == 8 { zero } else { e(tr.b.logical_not(tr.t.bool_, None, zero))? }
        }
        _ => return Err(format!("unknown conditional branch op {op}")),
    })
}

/// Loop targets of the innermost enclosing loop: (continue, merge).
type LoopLabels = Option<(Word, Word)>;

/// Whether any block in `nodes` accesses workgroup LDS (DS instructions other
/// than GDS and ds_swizzle).
fn uses_lds(code: &[Inst], ranges: &[std::ops::Range<usize>], nodes: &[Node]) -> bool {
    nodes.iter().any(|n| match n {
        Node::Block(b) => ranges[*b].clone().any(|k| {
            let i = &code[k];
            i.enc == crate::Enc::Ds && !i.ds_gds() && super::mem::ds_access_kind(i.op) != (false, false)
        }),
        Node::If { then_, else_, .. } => uses_lds(code, ranges, then_) || uses_lds(code, ranges, else_),
        Node::Loop { body, .. } => uses_lds(code, ranges, body),
        _ => false,
    })
}

/// At depth 0, before a construct whose blocks access LDS: a barrier if
/// accesses are pending (their order against the construct's is unknown),
/// and afterwards both kinds count as pending.
fn lds_construct(tr: &mut T, code: &[Inst], ranges: &[std::ops::Range<usize>], nodes: &[Node]) -> R<bool> {
    if tr.spec.stage != super::Stage::Compute || tr.lds.is_none() || tr.lds_base.is_some() || !uses_lds(code, ranges, nodes) {
        return Ok(false);
    }
    if tr.lds_phase != 0 {
        if tr.cf_depth == 0 {
            tr.workgroup_barrier()?;
        } else {
            tr.lds_hazards_nested += 1;
        }
    }
    Ok(true)
}

/// Emits `nodes`; at the end, unless the sequence ended in a jump, branches
/// to `after` (the enclosing construct's merge).
pub(crate) fn emit(tr: &mut T, code: &[Inst], ranges: &[std::ops::Range<usize>], nodes: &[Node], lp: LoopLabels, after: Option<Word>) -> R<()> {
    let ef = |x: rspirv::dr::Error| format!("{x:?}");
    for node in nodes {
        match node {
            Node::Block(b) => {
                for k in ranges[*b].clone() {
                    let ins = &code[k];
                    if ins.is_endpgm() || ins.branch_target().is_some() {
                        break;
                    }
                    translate_inst(tr, ins)?;
                }
            }
            Node::If { op, then_, else_, merged } => {
                if then_.is_empty() && else_.is_empty() {
                    continue; // both ways lead straight to the merge
                }
                let lds = lds_construct(tr, code, ranges, std::slice::from_ref(node))?;
                let c = branch_cond(tr, *op)?;
                let m = tr.b.id();
                let tl = if then_.is_empty() { m } else { tr.b.id() };
                let fl = if else_.is_empty() { m } else { tr.b.id() };
                tr.b.selection_merge(m, spirv::SelectionControl::NONE).map_err(ef)?;
                tr.b.branch_conditional(c, tl, fl, []).map_err(ef)?;
                tr.cf_depth += 1;
                if !then_.is_empty() {
                    e(tr.b.begin_block(Some(tl)))?;
                    emit(tr, code, ranges, then_, lp, Some(m))?;
                }
                if !else_.is_empty() {
                    e(tr.b.begin_block(Some(fl)))?;
                    emit(tr, code, ranges, else_, lp, Some(m))?;
                }
                tr.cf_depth -= 1;
                if lds {
                    tr.lds_phase = 3;
                }
                e(tr.b.begin_block(Some(m)))?;
                if !*merged {
                    tr.b.unreachable().map_err(ef)?;
                    return Ok(());
                }
            }
            Node::Loop { body, exits } => {
                let lds = lds_construct(tr, code, ranges, std::slice::from_ref(node))?;
                let (hdr, start, cont, merge) = (tr.b.id(), tr.b.id(), tr.b.id(), tr.b.id());
                tr.b.branch(hdr).map_err(ef)?;
                e(tr.b.begin_block(Some(hdr)))?;
                tr.b.loop_merge(merge, cont, spirv::LoopControl::NONE, []).map_err(ef)?;
                tr.b.branch(start).map_err(ef)?;
                e(tr.b.begin_block(Some(start)))?;
                tr.cf_depth += 1;
                emit(tr, code, ranges, body, Some((cont, merge)), None)?;
                tr.cf_depth -= 1;
                if lds {
                    tr.lds_phase = 3;
                }
                e(tr.b.begin_block(Some(cont)))?;
                tr.b.branch(hdr).map_err(ef)?;
                e(tr.b.begin_block(Some(merge)))?;
                if !*exits {
                    tr.b.unreachable().map_err(ef)?;
                    return Ok(());
                }
            }
            Node::Break => {
                let (_, merge) = lp.ok_or("break outside a loop")?;
                tr.b.branch(merge).map_err(ef)?;
                return Ok(());
            }
            Node::Continue => {
                let (cont, _) = lp.ok_or("continue outside a loop")?;
                tr.b.branch(cont).map_err(ef)?;
                return Ok(());
            }
            Node::Return => {
                super::mem::flush_dirty(tr)?;
                tr.b.ret().map_err(ef)?;
                return Ok(());
            }
        }
    }
    match after {
        Some(l) => tr.b.branch(l).map_err(ef),
        // (A loop body that ends without a jump continues; the top level returns.)
        None if lp.is_some() => tr.b.branch(lp.unwrap().0).map_err(ef),
        None => {
            super::mem::flush_dirty(tr)?;
            tr.b.ret().map_err(ef)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-dword SOPP programs: `('n')` s_nop, `('e')` s_endpgm, `('b', t)`
    /// s_branch to instruction t, `(op, t)` s_cbranch op to instruction t.
    fn program(ops: &[(u16, usize)]) -> Vec<Inst> {
        let words: Vec<u32> = ops
            .iter()
            .enumerate()
            .map(|(i, &(op, t))| {
                let simm = if op >= 2 { (t as i32 - i as i32 - 1) as i16 as u16 as u32 } else { 0 };
                0xBF80_0000 | ((op as u32) << 16) | simm
            })
            .collect();
        crate::decode_program(&words).unwrap()
    }
    const NOP: (u16, usize) = (0, 0);
    const END: (u16, usize) = (1, 0);

    fn plan_of(ops: &[(u16, usize)]) -> Result<Plan, String> {
        let code = program(ops);
        let starts = super::super::block_starts(&code);
        let ranges = super::super::block_ranges(&code, &starts);
        plan(&code, &starts, &ranges)
    }

    fn show(nodes: &[Node]) -> String {
        nodes
            .iter()
            .map(|n| match n {
                Node::Block(b) => format!("B{b}"),
                Node::If { op, then_, else_, merged } => format!("If{op}[{}|{}]{}", show(then_), show(else_), if *merged { "" } else { "!" }),
                Node::Loop { body, exits } => format!("Loop[{}]{}", show(body), if *exits { "" } else { "!" }),
                Node::Break => "brk".into(),
                Node::Continue => "cont".into(),
                Node::Return => "ret".into(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn if_then() {
        // 0: execz -> 3; 1, 2: body; 3: merge; 4: end
        let p = plan_of(&[(8, 3), NOP, NOP, NOP, END]).unwrap();
        assert_eq!(show(&p.nodes), "B0 If8[|B1] B2 ret");
    }

    #[test]
    fn if_then_else() {
        // 0: scc1 -> 3; 1: else; 2: branch -> 4; 3: then; 4: merge; 5: end
        let p = plan_of(&[(5, 3), NOP, (2, 4), NOP, NOP, END]).unwrap();
        assert_eq!(show(&p.nodes), "B0 If5[B2|B1] B3 ret");
    }

    #[test]
    fn nested_ifs_share_a_merge() {
        // 0: execz -> 4; 1: vccz -> 3; 2: inner body; 3: inner merge; 4: outer merge
        let p = plan_of(&[(8, 4), (6, 3), NOP, NOP, NOP, END]).unwrap();
        assert_eq!(show(&p.nodes), "B0 If8[|B1 If6[|B2] B3] B4 ret");
    }

    #[test]
    fn do_while_loop() {
        // 0: preheader; 1..2: body, 2: scc0 -> 1 (back edge); 3: end
        let p = plan_of(&[NOP, NOP, (4, 1), END]).unwrap();
        assert_eq!(show(&p.nodes), "B0 Loop[B1 If4[cont|brk]!] B2 ret");
        assert_eq!(p.loops, 1);
    }

    #[test]
    fn while_loop_with_break_and_continue() {
        // 1: header, execz -> 6 (exit); 2: vccz -> 5 (skip to latch);
        // 3: body; 4: falls into 5; 5: branch -> 1; 6: end
        let p = plan_of(&[NOP, (8, 6), (6, 5), NOP, NOP, (2, 1), END]).unwrap();
        assert_eq!(show(&p.nodes), "B0 Loop[B1 If8[brk|B2 If6[|B3] B4 cont]!] B5 ret");
    }

    #[test]
    fn early_return_nests_the_rest() {
        // 0: scc1 -> 3; 1: body; 2: end; 3: other path; 4: end
        let p = plan_of(&[(5, 3), NOP, END, NOP, END]).unwrap();
        assert_eq!(show(&p.nodes), "B0 If5[B2 ret|B1 ret]!");
    }

    #[test]
    fn block_reached_from_two_regions_is_duplicated() {
        // 0: scc1 -> 3 (T); 1 (F): vccz -> 4 (M), else 2 (X); 2: branch -> 4; 3: branch -> 2
        let p = plan_of(&[(5, 3), (6, 4), (2, 4), (2, 2), END]).unwrap();
        assert_eq!(show(&p.nodes), "B0 If5[B3 B2|B1 If6[|B2]] B4 ret");
    }

    #[test]
    fn shared_exit_after_an_early_return_is_duplicated() {
        // 0: scc1 -> 3; 1: vccz -> 6 (early end); 2: branch -> 4 (shared exit);
        // 3: falls into 4; 4, 5: exit sequence, end; 6: end
        let p = plan_of(&[(5, 3), (6, 6), (2, 4), NOP, NOP, END, END]).unwrap();
        assert_eq!(show(&p.nodes), "B0 If5[B3 B4 ret|B1 If6[B5 ret|B2 B4 ret]!]!");
    }

    #[test]
    fn loop_with_two_exits_is_refused() {
        // 1: header, scc1 -> 5 (exit A); 2: vccz -> 6 (exit B); 3: branch -> 1; 5, 6: ends
        assert!(plan_of(&[NOP, (5, 4), (6, 5), (2, 1), END, END]).is_err());
    }
}
