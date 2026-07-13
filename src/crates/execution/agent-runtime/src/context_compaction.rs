//! Pure policy helpers for context compaction watermarks.
//!
//! This module intentionally owns only threshold decisions. Product runtimes
//! decide what, if anything, can be compacted safely for a concrete transcript.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextCompactionTier {
    None,
    Snip,
    Prune,
    Summarize,
}

impl ContextCompactionTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Snip => "snip",
            Self::Prune => "prune",
            Self::Summarize => "summarize",
        }
    }

    pub fn is_local(self) -> bool {
        matches!(self, Self::Snip | Self::Prune)
    }
}

impl Default for ContextCompactionTier {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextCompactionPolicy {
    pub snip_ratio: f32,
    pub prune_ratio: f32,
    pub summarize_ratio: f32,
    pub protected_tail_tokens: usize,
    pub protected_recent_user_turns: usize,
    pub snip_preview_chars: usize,
    pub prune_preview_chars: usize,
    pub min_tool_result_chars: usize,
}

impl Default for ContextCompactionPolicy {
    fn default() -> Self {
        Self {
            snip_ratio: 0.60,
            prune_ratio: 0.80,
            summarize_ratio: 0.95,
            protected_tail_tokens: 8_000,
            protected_recent_user_turns: 2,
            snip_preview_chars: 1_200,
            prune_preview_chars: 240,
            min_tool_result_chars: 4_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextCompactionPlan {
    pub tier: ContextCompactionTier,
    pub pressure_ratio: f32,
}

impl ContextCompactionPolicy {
    pub fn plan(self, total_tokens: usize, input_limit: usize) -> ContextCompactionPlan {
        if input_limit == 0 {
            return ContextCompactionPlan {
                tier: ContextCompactionTier::None,
                pressure_ratio: 0.0,
            };
        }

        let pressure_ratio = total_tokens as f32 / input_limit as f32;
        let tier = if pressure_ratio >= self.summarize_ratio {
            ContextCompactionTier::Summarize
        } else if pressure_ratio >= self.prune_ratio {
            ContextCompactionTier::Prune
        } else if pressure_ratio >= self.snip_ratio {
            ContextCompactionTier::Snip
        } else {
            ContextCompactionTier::None
        };

        ContextCompactionPlan {
            tier,
            pressure_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ContextCompactionPolicy, ContextCompactionTier};

    #[test]
    fn policy_selects_expected_watermark_tiers() {
        let policy = ContextCompactionPolicy::default();

        assert_eq!(policy.plan(59, 100).tier, ContextCompactionTier::None);
        assert_eq!(policy.plan(60, 100).tier, ContextCompactionTier::Snip);
        assert_eq!(policy.plan(80, 100).tier, ContextCompactionTier::Prune);
        assert_eq!(policy.plan(95, 100).tier, ContextCompactionTier::Summarize);
    }

    #[test]
    fn policy_disables_compaction_without_input_limit() {
        let policy = ContextCompactionPolicy::default();

        assert_eq!(policy.plan(100, 0).tier, ContextCompactionTier::None);
        assert_eq!(policy.plan(100, 0).pressure_ratio, 0.0);
    }
}
