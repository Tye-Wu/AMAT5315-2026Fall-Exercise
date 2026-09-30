use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ActionRecord {
    pub action: String,
    #[serde(rename = "step")]
    pub n: usize,
    pub saved_states: usize,
}

fn binomial_fit(n: usize, delta: usize) -> usize {
    let mut tau = 1_usize;
    while choose(tau + delta, tau) < n {
        tau += 1;
    }
    tau
}

fn choose(n: usize, k: usize) -> usize {
    let k = k.min(n.saturating_sub(k));
    let mut value = 1_u128;
    for i in 0..k {
        value = value * (n - i) as u128 / (i + 1) as u128;
        if value > usize::MAX as u128 {
            return usize::MAX;
        }
    }
    value as usize
}

fn mid(delta: usize, tau: usize, sigma: usize, phi: usize) -> usize {
    let den = delta + tau;
    let num = delta * sigma + tau * phi;
    let mut kappa = num.div_ceil(den);
    if kappa >= phi && delta > 0 {
        kappa = (sigma + 1).max(phi.saturating_sub(1));
    }
    kappa
}

struct Planner {
    actions: Vec<ActionRecord>,
    saved: Vec<usize>,
    peak: usize,
    limit: usize,
}

impl Planner {
    fn action(&mut self, action: &str, n: usize) {
        self.actions.push(ActionRecord {
            action: action.into(),
            n,
            saved_states: self.saved.len(),
        });
    }

    fn recurse(
        &mut self,
        mut delta: usize,
        mut tau: usize,
        beta: usize,
        sigma: usize,
        mut phi: usize,
    ) -> Result<(), String> {
        if sigma > beta {
            delta = delta
                .checked_sub(1)
                .ok_or("Treeverse attempted to exceed checkpoint budget")?;
            if !self.saved.contains(&beta) {
                return Err(format!("missing saved replay origin s_{beta}"));
            }
            self.action("restore", beta);
            for j in beta..sigma {
                self.action("call", j);
            }
            if self.saved.contains(&sigma) {
                return Err(format!("state s_{sigma} stored twice"));
            }
            self.saved.push(sigma);
            self.peak = self.peak.max(self.saved.len());
            if self.saved.len() > self.limit {
                return Err(format!(
                    "checkpoint budget exceeded: {} > {}",
                    self.saved.len(),
                    self.limit
                ));
            }
            self.action("store", sigma);
        } else if sigma < beta {
            return Err("Treeverse interval is not ordered".into());
        }

        let mut kappa = mid(delta, tau, sigma, phi);
        while tau > 0 && kappa < phi {
            self.recurse(delta, tau, sigma, kappa, phi)?;
            tau -= 1;
            phi = kappa;
            kappa = mid(delta, tau, sigma, phi);
        }
        if !self.saved.contains(&sigma) {
            return Err(format!("missing reverse input s_{sigma}"));
        }
        self.action("grad", sigma);
        if sigma > beta {
            let pos = self.saved.iter().position(|&n| n == sigma).unwrap();
            self.saved.remove(pos);
            self.action("fetch", sigma);
        }
        Ok(())
    }
}

pub fn treeverse_schedule(
    steps: usize,
    extra_slots: usize,
) -> Result<(Vec<ActionRecord>, usize), String> {
    if steps == 0 || extra_slots == 0 {
        return Err("Treeverse requires positive steps and at least one extra slot".into());
    }
    let limit = extra_slots + 1;
    let mut planner = Planner {
        actions: Vec::new(),
        saved: vec![0],
        peak: 1,
        limit,
    };
    let tau = binomial_fit(steps, extra_slots);
    planner.recurse(extra_slots, tau, 0, 0, steps)?;
    if planner.saved != vec![0] {
        return Err(format!(
            "Treeverse left unexpected saved states: {:?}",
            planner.saved
        ));
    }
    Ok((planner.actions, planner.peak))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_course_reference_work_and_state_peaks() {
        let expected = [(1, 28_680, 2), (3, 1_695, 4), (5, 990, 6), (10, 642, 11)];
        for (budget, calls, peak) in expected {
            let (actions, found_peak) = treeverse_schedule(240, budget).unwrap();
            assert_eq!(actions.iter().filter(|a| a.action == "call").count(), calls);
            assert_eq!(found_peak, peak);
            let grads: Vec<_> = actions
                .iter()
                .filter(|a| a.action == "grad")
                .map(|a| a.n)
                .collect();
            assert_eq!(grads, (0..240).rev().collect::<Vec<_>>());
        }
        let (actions, peak) = treeverse_schedule(6, 2).unwrap();
        assert_eq!(actions.iter().filter(|a| a.action == "call").count(), 8);
        assert_eq!(peak, 3);
    }

    #[test]
    fn replayed_primal_states_give_the_same_reverse_derivative_as_full_history() {
        let n = 6;
        let (actions, _) = treeverse_schedule(n, 2).unwrap();
        let step = |x: f64| x * x + 1.0;
        let mut full = vec![0.1_f64];
        for _ in 0..n {
            full.push(step(*full.last().unwrap()));
        }
        let mut expected = 1.0;
        for k in (0..n).rev() {
            expected *= 2.0 * full[k];
        }

        let mut saved = std::collections::BTreeMap::from([(0_usize, 0.1_f64)]);
        let mut working = None::<(usize, f64)>;
        let mut actual = 1.0;
        for a in actions {
            match a.action.as_str() {
                "restore" => working = Some((a.n, *saved.get(&a.n).unwrap())),
                "call" => {
                    let (k, x) = working.take().unwrap();
                    assert_eq!(k, a.n);
                    working = Some((k + 1, step(x)));
                }
                "store" => {
                    let (k, x) = working.unwrap();
                    assert_eq!(k, a.n);
                    assert!(saved.insert(k, x).is_none());
                }
                "grad" => actual *= 2.0 * saved[&a.n],
                "fetch" => {
                    assert!(saved.remove(&a.n).is_some());
                }
                other => panic!("unexpected action {other}"),
            }
            assert_eq!(saved.len(), a.saved_states);
        }
        assert!((actual - expected).abs() < 1e-12);
        assert_eq!(saved.keys().copied().collect::<Vec<_>>(), vec![0]);
    }
}
