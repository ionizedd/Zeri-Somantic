/*
 * zeri_agent.hpp  —  Reinforcement Learning extension for zeri_brain
 * ==================================================================
 * Version : 0.4.0  |  Author: Zeri / ionizedd  |  License: MIT
 *
 * Extends zeri_brain.hpp with action learning:
 *
 *   • Epsilon-greedy exploration (try random things early, exploit later)
 *   • Eligibility traces       (credit past actions for delayed rewards)
 *   • Two-layer SNN            (hidden features → action logits)
 *   • Reward-modulated STDP    (three-factor: pre × post × reward)
 *   • Actor-Critic with TD(λ)  (critic bootstraps value, actor uses advantage)
 *   • Genetic breeding         (cross two trained agents → smarter offspring)
 *   • Clone / snapshot         (fork an agent mid-training, rollback-ready)
 *   • Tournament selection     (evolve a population with minimal code)
 *
 * The personality-from-seed angle:
 *   Different seeds crystallize into different PLAY STYLES.
 *   A "bold" seed explores aggressively. A "cautious" seed plays safe.
 *   Breed two agents → offspring inherits a hybrid strategy AND starts ahead.
 *
 * Genetic RL in 3 lines:
 *   ZeriAgent* a = zeri_agent_from_name("alice", 16, 4, 64, 1.0f);
 *   ZeriAgent* b = zeri_agent_from_name("bob",   16, 4, 64, 1.0f);
 *   // ... train a and b ...
 *   ZeriAgent* child = zeri_agent_breed(a, b, "child", 0.5f, 0.3f);
 *   // child starts with blended experience of both parents
 *
 * Usage:
 *   #define ZERI_IMPLEMENTATION
 *   #define ZERI_AGENT_IMPLEMENTATION
 *   #include "zeri_brain.hpp"
 *   #include "zeri_agent.hpp"
 *
 *   ZeriAgent* agent = zeri_agent_create(seed, state_dim, n_actions, hidden_dim, epsilon);
 *   int action = zeri_agent_act(agent, state_vec, state_dim);
 *   zeri_agent_reinforce(agent, state_vec, state_dim, action, next_state, reward, done);
 *   zeri_agent_decay_epsilon(agent);   // call each episode
 *   zeri_agent_destroy(agent);
 */

#pragma once
#ifndef ZERI_AGENT_H
#define ZERI_AGENT_H

#ifndef SOMATIC_API
#  if defined(_WIN32) || defined(__CYGWIN__)
#    define SOMATIC_API __declspec(dllexport)
#  else
#    define SOMATIC_API __attribute__((visibility("default")))
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZeriAgent ZeriAgent;

/* =========================================================================
   PUBLIC API
   ========================================================================= */

/*
 * zeri_agent_create
 *   seed       : 128-byte personality seed
 *   state_dim  : dimensionality of state vector (must be multiple of 8)
 *   n_actions  : number of discrete actions
 *   hidden_dim : hidden layer size (0 = skip hidden layer, direct state→action)
 *   epsilon    : initial exploration rate (1.0 = pure random, 0.0 = pure greedy)
 */
SOMATIC_API ZeriAgent* zeri_agent_create(const uint8_t* seed,
                              int state_dim, int n_actions, int hidden_dim,
                              float epsilon);

/*
 * zeri_agent_act
 *   Returns the chosen action index [0, n_actions).
 *   Explores (random) with probability epsilon, exploits (argmax) otherwise.
 *   Call this every step to get the agent's decision.
 */
SOMATIC_API int zeri_agent_act(ZeriAgent* agent, const float* state, int state_dim);

/*
 * zeri_agent_act_greedy
 *   Always returns argmax — no exploration.
 *   Use this after training is done, for evaluation.
 */
SOMATIC_API int zeri_agent_act_greedy(ZeriAgent* agent, const float* state, int state_dim);

/*
 * zeri_agent_act_stochastic_temp
 *   Samples an action directly from the actor's own policy distribution
 *   (softmax over its logits at the given temperature) instead of either
 *   epsilon-greedy's blind uniform-random flip or act_greedy's hard,
 *   deterministic argmax. temperature=1.0 samples the policy exactly as
 *   the actor currently believes it (its "honest" distribution);
 *   temperature<1.0 sharpens toward argmax; temperature>1.0 flattens
 *   toward uniform.
 *
 *   Why this is more "natural" than epsilon-greedy: the actor's
 *   distribution is already entropy-regularized (see ZAGENT_ENTROPY_BETA
 *   in zeri_agent_reinforce) specifically so it never fully collapses to
 *   a one-hot certainty — there is always live probability mass on
 *   near-tied alternatives. Sampling from that distribution means
 *   exploration intensity tracks the agent's own uncertainty (directly
 *   measurable via zeri_agent_policy_entropy) instead of an external,
 *   uncertainty-blind epsilon schedule. Two concrete side effects: (1) a
 *   converged policy with a near-tied 2-cycle at some state keeps
 *   sampling both sides indefinitely instead of ever fully locking into
 *   one, so it can't get stuck in an infinite loop the way act_greedy
 *   can; (2) it still explores less over time as training sharpens real
 *   preferences, without a separately scheduled decay to tune.
 *
 *   Purely additive: does not alter zeri_agent_act or
 *   zeri_agent_act_greedy, and calling it never changes agent state
 *   beyond what any forward pass already does (SNN membrane potentials).
 */
SOMATIC_API int zeri_agent_act_stochastic_temp(ZeriAgent* agent, const float* state,
                                    int state_dim, float temperature);

/* zeri_agent_act_stochastic — convenience wrapper for temperature=1.0. */
SOMATIC_API int zeri_agent_act_stochastic(ZeriAgent* agent, const float* state, int state_dim);

/*
 * zeri_agent_reinforce
 *   Present the (state, action, next_state, reward, is_terminal) tuple.
 *   Uses Temporal Difference (TD) learning for Actor-Critic updates.
 */
SOMATIC_API void zeri_agent_reinforce(ZeriAgent* agent,
                           const float* state, int state_dim,
                           int action,
                           const float* next_state,
                           float reward,
                           int is_terminal);

/* Clear eligibility traces at start of a new episode. */
SOMATIC_API void zeri_agent_reset_traces(ZeriAgent* agent);

/*
 * zeri_agent_decay_epsilon
 *   Call once per episode to reduce exploration over time.
 *   epsilon → epsilon * decay_rate (default 0.995), clamped at epsilon_min.
 */
SOMATIC_API void zeri_agent_decay_epsilon(ZeriAgent* agent);

/* Set custom decay rate and minimum (defaults: rate=0.995, min=0.05) */
SOMATIC_API void zeri_agent_set_decay(ZeriAgent* agent, float rate, float min_eps);

/*
 * zeri_agent_set_forget_rate
 *   Controls the per-step magnitude of "Biological Synaptic Forgetting"
 *   (see zeri_agent_reinforce) -- the L1 pull of every weight delta back
 *   toward its seed baseline, applied on any step with nonzero TD error.
 *   Default is 0.00002f (unchanged from the original fixed-rate behavior;
 *   existing callers that never call this setter see byte-identical
 *   training to before this API existed).
 *
 *   Why this needed to become configurable: the rate is a flat per-step
 *   constant, but lr_actor/lr_hidden/lr_critic all decay toward floors via
 *   zeri_agent_decay_epsilon() (see ZAGENT comment there). Once the
 *   learning rate has decayed far below its starting value but the forget
 *   rate hasn't moved at all, forgetting can erode a rare, hard-won
 *   correct weight pattern faster than the shrunken learning rate can
 *   reinforce it -- measured directly on a long two-chokepoint maze
 *   (grid_demo.c course preset 3): with forgetting at its default rate,
 *   the agent kept re-discovering and then re-forgetting the correct path
 *   and never converged even after 8000 episodes; setting the rate to 0.0f
 *   (full disable) let the same agent converge cleanly by episode ~6000.
 *   Pass 0.0f to disable forgetting entirely, or any other rate to tune
 *   the tradeoff for a specific task. See dev/CHANGELOG.md for the full
 *   before/after measurements. */
SOMATIC_API void zeri_agent_set_forget_rate(ZeriAgent* agent, float rate);

/*
 * zeri_agent_set_forget_surprise_gate
 *   Switches Biological Synaptic Forgetting from its default instantaneous
 *   gate to a rolling-average one. Pass a negative threshold to restore the
 *   default (instantaneous |td_error| > 1e-4f, unchanged from before this
 *   API existed); pass a threshold >= 0.0f to gate on a rolling EMA of
 *   |td_error| instead (see zeri_agent_get_avg_abs_td), with a one-way
 *   latch: once that average drops below the threshold, forgetting locks
 *   off PERMANENTLY for this agent, rather than re-checking every step.
 *
 *   Why the instantaneous gate needed replacing: `fabsf(td_error) > 1e-4f`
 *   reads as "only forget when surprised," but td_error is a continuous
 *   float -- landing within 1e-4 of exactly zero essentially never happens
 *   in practice, so the gate fires on nearly every step regardless of
 *   whether the agent is actually struggling. It wasn't selective at all.
 *
 *   Why a plain (non-latching) rolling-average gate wasn't enough: tested
 *   first without the latch, toggling forgetting on/off every step based on
 *   whether the rolling average was currently above or below threshold.
 *   That was unreliable -- a single re-trigger late in training, even from
 *   a state where the gate was closed almost the entire time, could send
 *   the whole run down a measurably worse trajectory (this training
 *   process is sensitive enough to a single weight perturbation that
 *   "almost always closed" isn't the same as "closed"). The one-way latch
 *   fixes that: once earned, the lock can't be undone by a later transient
 *   spike, closer to real synaptic consolidation than a chattering toggle.
 *
 *   Why this is better than zeri_agent_set_forget_rate(agent, 0.0f) (full,
 *   permanent disable from step one): this keeps forgetting genuinely
 *   active for as long as the agent is actually struggling (rolling average
 *   error still high) and only earns the lock-off once behavior has
 *   settled. On an easy task it'll latch off almost immediately and behave
 *   like full disable; on a harder or later-converging task it keeps doing
 *   real forgetting work for longer before consolidating. Measured on
 *   grid_demo.c course preset 3 with threshold=0.3f: converges cleanly by
 *   episode 8000, same as full disable, with zero regression on presets 0,
 *   1, 2, 5. See dev/CHANGELOG.md for the full investigation. */
SOMATIC_API void zeri_agent_set_forget_surprise_gate(ZeriAgent* agent, float threshold);

/* Rolling EMA (~100-step time constant) of |td_error| -- "how surprised
 * this agent has been lately." Always maintained regardless of which
 * forgetting gate is active; read this to pick a sensible threshold for
 * zeri_agent_set_forget_surprise_gate on a new task. */
SOMATIC_API float zeri_agent_get_avg_abs_td(ZeriAgent* agent);

/* Get action probabilities over all actions for a given state */
SOMATIC_API void zeri_agent_probs(ZeriAgent* agent, const float* state, int state_dim,
                      float* out_probs);

/* Stats: [epsilon, episodes, total_reward, avg_reward_100, steps] */
SOMATIC_API void zeri_agent_stats(ZeriAgent* agent, float* out_5);

/* Save/load agent memory (delta from seed state) */
SOMATIC_API int  zeri_agent_consolidate(ZeriAgent* agent, uint8_t* buf, int buf_size);
SOMATIC_API int  zeri_agent_load_memory(ZeriAgent* agent, const uint8_t* buf, int buf_size);

SOMATIC_API void zeri_agent_destroy(ZeriAgent* agent);

/* Named agent — same name → same starting play style */
SOMATIC_API ZeriAgent* zeri_agent_from_name(const char* name,
                                  int state_dim, int n_actions, int hidden_dim,
                                  float epsilon);

/* -------------------------------------------------------------------------
 * GENETIC / BREEDING API
 * -------------------------------------------------------------------------
 *
 * zeri_agent_breed
 *   Creates an offspring by blending the LEARNED DELTAS of two parents.
 *   Both parents must share the same architecture (state_dim, n_actions,
 *   hidden_dim). The offspring gets its own base personality from
 *   offspring_name, then inherits:
 *
 *     W_child = W0_child + alpha*(Wa-Wa0) + (1-alpha)*(Wb-Wb0)
 *
 *   Where Wa0/Wb0 are each parent's seed-baseline weights.
 *   This models epigenetic inheritance: nature (new seed) + nurture (experience).
 *
 *   alpha=0.5 → equal mix | alpha=1.0 → pure parent_a | alpha=0.0 → pure parent_b
 *   epsilon   → starting exploration rate for the child
 *
 *   The child immediately starts AHEAD of either cold-start parent because
 *   it inherits the beneficial weight adjustments without the bad early
 *   exploration that discovered them.
 */
SOMATIC_API ZeriAgent* zeri_agent_breed(const ZeriAgent* parent_a,
                             const ZeriAgent* parent_b,
                             const char* offspring_name,
                             float alpha,
                             float epsilon);

/*
 * zeri_agent_breed_seed
 *   Same as zeri_agent_breed but offspring personality comes from a raw 128-byte seed.
 */
SOMATIC_API ZeriAgent* zeri_agent_breed_seed(const ZeriAgent* parent_a,
                                  const ZeriAgent* parent_b,
                                  const uint8_t* offspring_seed,
                                  float alpha,
                                  float epsilon);

/*
 * zeri_agent_clone
 *   Deep copy of an agent including all trained weights and stats.
 *   Use for: branching training runs, rollback, ensemble evaluation.
 */
SOMATIC_API ZeriAgent* zeri_agent_clone(const ZeriAgent* src);

/*
 * zeri_agent_fitness
 *   Returns a single scalar representing agent quality.
 *   Currently: avg_reward_100 / max_steps — a normalized performance metric.
 *   Pass max_steps_per_ep (e.g. 80) to normalize.
 *   Use this for tournament selection: keep top-k of a population.
 */
SOMATIC_API float zeri_agent_fitness(ZeriAgent* ag, int max_steps_per_ep);

/*
 * zeri_agent_get_delta_norm
 *   Returns the L1 norm of all weight deltas from seed baseline.
 *   Measures how much an agent has "drifted" from its starting personality.
 *   High delta_norm = heavily specialized; low = still close to instinct.
 */
SOMATIC_API float zeri_agent_get_delta_norm(const ZeriAgent* ag);

/*
 * zeri_agent_mutate
 *   Adds small Gaussian noise to the trained weight deltas (not the seed).
 *   Simulates genetic mutation. Useful after breeding to maintain diversity.
 *   sigma: standard deviation of noise (try 0.02 to 0.1)
 */
SOMATIC_API void zeri_agent_mutate(ZeriAgent* ag, float sigma);

/*
 * zeri_agent_value
 *   Returns the critic's estimated value for a given state.
 *   Useful for: debugging policy quality, identifying "scary" states, viz.
 */
SOMATIC_API float zeri_agent_value(ZeriAgent* ag, const float* state, int state_dim);

/*
 * zeri_agent_policy_entropy
 *   Returns the Shannon entropy of the policy over all actions for a state.
 *   High entropy = uncertain/exploring. Low entropy = confident/committed.
 *   Range: [0, log(n_actions)]. Use to detect premature convergence.
 */
SOMATIC_API float zeri_agent_policy_entropy(ZeriAgent* ag, const float* state, int state_dim);

/*
 * zeri_agent_reset_stats
 *   Zero out episode/step counters and reward window without changing weights.
 *   Use when resuming training after breeding to get clean metrics.
 */
SOMATIC_API void zeri_agent_reset_stats(ZeriAgent* ag);

#ifdef __cplusplus
}
#endif


/* =========================================================================
   IMPLEMENTATION
   ========================================================================= */
#ifdef ZERI_AGENT_IMPLEMENTATION

#if !defined(ZERI_NO_AVX2) && defined(__AVX2__)
  #include <immintrin.h>
  #define ZAGENT_AVX2 1
#endif

#define ZAGENT_TRACE_DECAY  0.92f   /* eligibility trace decay per step */
#define ZAGENT_LR_ACTOR     0.015f  /* learning rate for actor weights */
#define ZAGENT_LR_HIDDEN    0.008f  /* learning rate for hidden layer */
#define ZAGENT_CLIP         5.0f    /* weight clamp */
#define ZAGENT_STDP_A       0.012f  /* STDP amplitude */
#define ZAGENT_ENTROPY_BETA 0.05f   /* entropy-regularization strength (prevents policy saturation/vanishing-gradient collapse -- see zeri_agent_reinforce) */
#define ZAGENT_LOGIT_CLIP   1.5f   /* hard clamp on actor logits before softmax -- bounds max action-prob gap regardless of accumulated weight magnitude (see zagent__actor_fwd) */

/* ── fast dot product (reuse from zeri_brain) ──────────────────────────── */

static inline uint64_t zagent__sm64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

#ifdef ZAGENT_AVX2
static inline float zagent__dot(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#else
static inline float zagent__dot(const float* a, const float* b, int n) {
    float s = 0.f; for (int i=0;i<n;i++) s+=a[i]*b[i]; return s;
}
#endif

static void zagent__softmax(const float* lg, float* pr, int n) {
    float mx = lg[0]; for (int i=1;i<n;i++) if(lg[i]>mx) mx=lg[i];
    float s=0; for(int i=0;i<n;i++){pr[i]=expf(lg[i]-mx);s+=pr[i];}
    for(int i=0;i<n;i++) pr[i]/=s;
}

static inline float zagent__clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ── seed → initial ternary weights ───────────────────────────────────── */

static void zagent__vfield(const uint64_t* s16, int start, int count, float* out) {
    int blk = start/8, off = start%8, i = 0;
    uint64_t cur = zagent__sm64(s16[blk&15] ^ zagent__sm64((uint64_t)blk));
    uint8_t  b[8]; memcpy(b, &cur, 8);
    for (; off<8 && i<count; off++,i++) out[i] = (float)((int)(b[off]%3)-1);
    blk++;
    while (i+8<=count) {
        cur = zagent__sm64(s16[blk&15] ^ zagent__sm64((uint64_t)blk));
        memcpy(b,&cur,8);
        for(int j=0;j<8;j++) out[i+j]=(float)((int)(b[j]%3)-1);
        i+=8; blk++;
    }
    if(i<count){
        cur=zagent__sm64(s16[blk&15]^zagent__sm64((uint64_t)blk));
        memcpy(b,&cur,8);
        for(int j=0;i<count;j++,i++) out[i]=(float)((int)(b[j]%3)-1);
    }
}

/* ── Agent struct ─────────────────────────────────────────────────────── */

struct ZeriAgent {
    int      state_dim;     /* input dimension */
    int      n_actions;     /* output dimension */
    int      hidden_dim;    /* 0 = no hidden layer */
    uint64_t seed16[16];

    /* Hidden layer (optional): state → hidden */
    float*   Wh;            /* [hidden_dim × state_dim] */
    float*   Wh0;           /* seed snapshot */
    float*   bh;            /* [hidden_dim] */
    float*   hidden_act;    /* [hidden_dim] activation buffer */
    float*   last_hidden_act; /* [hidden_dim] previous step activation */
    float*   membrane_pot;  /* [hidden_dim] SNN membrane potential */
    float*   Wh_rec;        /* [hidden_dim × hidden_dim] recurrent feedback */
    float*   Wh_rec0;       /* seed snapshot */

    /* Actor layer: hidden (or state) → action logits */
    int      actor_in;      /* = hidden_dim or state_dim */
    float*   Wa;            /* [n_actions × actor_in] */
    float*   Wa0;
    float*   ba;            /* [n_actions] */

    /* Critic layer: actor_in → state value */
    float*   Wc;            /* [actor_in] */
    float*   Wc0;           /* seed snapshot */
    float    bc;            /* critic bias */
    float    gamma;         /* discount factor */
    float    lr_critic;     /* critic learning rate */
    float    lr_actor;      /* actor learning rate */
    float    lr_hidden;     /* hidden layer learning rate */

    /* Eligibility traces — decaying memory of recent (state,action) gradient */
    float*   elig_Wh;       /* [hidden_dim × state_dim] */
    float*   elig_Wh_rec;   /* [hidden_dim × hidden_dim] */
    float*   elig_Wa;       /* [n_actions × actor_in] */
    float*   elig_bh;       /* [hidden_dim] */
    float*   elig_ba;       /* [n_actions] */

    /* STDP pre-synaptic traces */
    float*   pre_h;         /* [state_dim] */
    float*   pre_a;         /* [actor_in] */

    /* Exploration */
    float    epsilon;
    float    eps_decay;
    float    eps_min;
    uint64_t rng_state;
    float    forget_rate;   /* Biological Synaptic Forgetting rate; see zeri_agent_set_forget_rate */
    float    avg_abs_td;    /* rolling EMA of |td_error|; see zeri_agent_get_avg_abs_td */
    float    forget_surprise_threshold; /* <0 = instantaneous gate (default); >=0 = rolling-average gate; see zeri_agent_set_forget_surprise_gate */
    int      forget_locked_off; /* one-way latch for the rolling-average gate */

    /* Stats */
    uint64_t steps;
    uint64_t episodes;
    double   total_reward;
    float    reward_window[100];
    int      reward_win_idx;
};

/* ── RNG ────────────────────────────────────────────────────────────────── */

static inline float zagent__randf(ZeriAgent* ag) {
    ag->rng_state = zagent__sm64(ag->rng_state + 1);
    return (float)((ag->rng_state >> 11) & 0x1FFFFF) / (float)0x1FFFFF;
}
static inline int zagent__randi(ZeriAgent* ag, int n) {
    ag->rng_state = zagent__sm64(ag->rng_state + 1);
    return (int)(ag->rng_state % (uint64_t)n);
}

/* ── Forward pass ────────────────────────────────────────────────────────── */

/*
 * Compute hidden activations with recurrent feedback and leaky SNN dynamics.
 */
static void zagent__hidden_fwd(ZeriAgent* ag, const float* state) {
    if (!ag->hidden_dim) return;
    int H = ag->hidden_dim, S = ag->state_dim;
    memcpy(ag->last_hidden_act, ag->hidden_act, (size_t)H * sizeof(float));
    for (int h = 0; h < H; h++) {
        float I_input = zagent__dot(ag->Wh + h*S, state, S) + ag->bh[h];
        float I_rec = zagent__dot(ag->Wh_rec + h*H, ag->last_hidden_act, H);
        
        // Leaky Spiking Integrator Dynamics
        ag->membrane_pot[h] = ag->membrane_pot[h] * 0.85f + I_input + I_rec;
        
        if (ag->membrane_pot[h] >= 1.0f) {
            ag->hidden_act[h] = 1.0f; // Spike!
            ag->membrane_pot[h] = 0.0f; // Reset!
        } else {
            ag->hidden_act[h] = 0.0f; // No spike
            if (ag->membrane_pot[h] < -1.0f) ag->membrane_pot[h] = -1.0f;
        }
    }
}

/* Actor forward → logits and probs */
static void zagent__actor_fwd(ZeriAgent* ag, const float* actor_in,
                               float* logits, float* probs) {
    int A = ag->n_actions, D = ag->actor_in;
    for (int a = 0; a < A; a++) {
        logits[a] = zagent__dot(ag->Wa + a*D, actor_in, D) + ag->ba[a];
        /* Logit clamp: per-weight clamps (ZAGENT_CLIP on Wa/ba) bound any
         * SINGLE synapse, but the dot product sums over D (up to 64+)
         * inputs, so the summed logit can still grow far larger than any
         * individual weight -- enough to saturate the softmax to float32's
         * 0.0/1.0 limits. Once there, both the policy-gradient term
         * (1_chosen - prob) AND the entropy-regularization gradient
         * vanish identically (both are proportional to p*(1-p)-like
         * quantities), so NOTHING can recover the policy afterward.
         * Clamping the logit directly bounds the gap between actions
         * regardless of how large the underlying weighted sum becomes,
         * keeping a live, recoverable gradient always available. */
        logits[a] = zagent__clamp(logits[a], -ZAGENT_LOGIT_CLIP, ZAGENT_LOGIT_CLIP);
    }
    zagent__softmax(logits, probs, A);
}

/* Full forward: state → probs */
static void zagent__forward(ZeriAgent* ag, const float* state,
                             float* logits, float* probs) {
    zagent__hidden_fwd(ag, state);
    const float* actor_in = ag->hidden_dim ? ag->hidden_act : state;
    zagent__actor_fwd(ag, actor_in, logits, probs);
}

/* ── Eligibility trace update ─────────────────────────────────────────── */

/*
 * After choosing action `a`, update eligibility traces to record
 * "these weights were responsible for this action."
 * Traces decay over time — older decisions get less credit.
 */
static void zagent__update_eligibility(ZeriAgent* ag,
                                        const float* state,
                                        const float* actor_in_vec,
                                        const float* probs,
                                        int chosen_action) {
    int A = ag->n_actions, D = ag->actor_in, S = ag->state_dim;

    /* Decay all existing traces */
    for (int i = 0; i < A * D; i++) ag->elig_Wa[i] *= ZAGENT_TRACE_DECAY;
    for (int i = 0; i < A;     i++) ag->elig_ba[i] *= ZAGENT_TRACE_DECAY;
    if (ag->hidden_dim) {
        int H = ag->hidden_dim;
        for (int i = 0; i < H * S; i++) ag->elig_Wh[i] *= ZAGENT_TRACE_DECAY;
        for (int i = 0; i < H * H; i++) ag->elig_Wh_rec[i] *= ZAGENT_TRACE_DECAY;
        for (int i = 0; i < H;     i++) ag->elig_bh[i] *= ZAGENT_TRACE_DECAY;
    }

    /* Accumulate gradient for chosen action into traces */
    /* grad_a[a] = (1_a==chosen - prob_a) — policy gradient */
    for (int a = 0; a < A; a++) {
        float g = (a == chosen_action ? 1.0f : 0.0f) - probs[a];
        float* Wa_row = ag->elig_Wa + a * D;
        for (int d = 0; d < D; d++) Wa_row[d] += g * actor_in_vec[d];
        ag->elig_ba[a] += g;
    }

    /* Hidden layer gradient (backprop through threshold & recurrent connections) */
    if (ag->hidden_dim) {
        int H = ag->hidden_dim;
        for (int h = 0; h < H; h++) {
            if (ag->hidden_act[h] <= 0.0f) continue;  /* Spiked gate */
            float dh = 0.0f;
            for (int a = 0; a < A; a++) {
                float g = (a == chosen_action ? 1.0f : 0.0f) - probs[a];
                dh += g * ag->Wa[a * D + h];
            }
            
            float* Wh_row = ag->elig_Wh + h * ag->state_dim;
            for (int s = 0; s < ag->state_dim; s++)
                Wh_row[s] += dh * state[s];
                
            float* Wh_rec_row = ag->elig_Wh_rec + h * H;
            for (int h2 = 0; h2 < H; h2++)
                Wh_rec_row[h2] += dh * ag->last_hidden_act[h2];
                
            ag->elig_bh[h] += dh;
        }
    }
}

/* ── Reward-modulated STDP weight update ─────────────────────────────── */

static void zagent__apply_reward(ZeriAgent* ag, float reward) {
    int A = ag->n_actions, D = ag->actor_in;
    float lr = ag->lr_actor * reward;

    /* Actor weights: W += lr × eligibility_trace */
    for (int i = 0; i < A * D; i++) {
        ag->Wa[i] += lr * ag->elig_Wa[i];
        ag->Wa[i]  = zagent__clamp(ag->Wa[i], -ZAGENT_CLIP, ZAGENT_CLIP);
    }
    for (int i = 0; i < A; i++) {
        ag->ba[i] += lr * ag->elig_ba[i];
        /* Bias clamp: without this, a heavily and consistently reinforced
         * action's bias can grow unbounded over thousands of steps (unlike
         * Wa above, this was never clamped) -- producing a logit gap so
         * large the softmax saturates to the limits of float precision,
         * which is what defeats entropy regularization too (its gradient
         * also vanishes at full saturation). Bounding ba keeps the gap
         * recoverable. */
        ag->ba[i] = zagent__clamp(ag->ba[i], -ZAGENT_CLIP, ZAGENT_CLIP);
    }

    /* Hidden layer */
    if (ag->hidden_dim) {
        int H = ag->hidden_dim, S = ag->state_dim;
        float lr_h = ag->lr_hidden * reward;
        for (int i = 0; i < H*S; i++) {
            ag->Wh[i] += lr_h * ag->elig_Wh[i];
            ag->Wh[i]  = zagent__clamp(ag->Wh[i], -ZAGENT_CLIP, ZAGENT_CLIP);
        }
        for (int i = 0; i < H*H; i++) {
            ag->Wh_rec[i] += lr_h * ag->elig_Wh_rec[i];
            ag->Wh_rec[i] = zagent__clamp(ag->Wh_rec[i], -ZAGENT_CLIP, ZAGENT_CLIP);
        }
        for (int i = 0; i < H; i++) {
            ag->bh[i] += lr_h * ag->elig_bh[i];
            ag->bh[i] = zagent__clamp(ag->bh[i], -ZAGENT_CLIP, ZAGENT_CLIP);
        }
    }
}

/* =========================================================================
   PUBLIC FUNCTIONS
   ========================================================================= */

ZeriAgent* zeri_agent_create(const uint8_t* seed,
                              int state_dim, int n_actions, int hidden_dim,
                              float epsilon) {
    /* align dims to 8 for AVX2 */
    if (state_dim  % 8) state_dim  = (state_dim  + 7) & ~7;
    if (hidden_dim % 8) hidden_dim = (hidden_dim + 7) & ~7;

    ZeriAgent* ag = (ZeriAgent*)calloc(1, sizeof(ZeriAgent));
    ag->state_dim  = state_dim;
    ag->n_actions  = n_actions;
    ag->hidden_dim = hidden_dim;
    ag->actor_in   = hidden_dim ? hidden_dim : state_dim;
    ag->epsilon    = epsilon;
    ag->eps_decay  = 0.995f;
    ag->eps_min    = 0.05f;
    ag->forget_rate = 0.00002f; /* preserves original fixed-rate forgetting behavior by default */
    ag->avg_abs_td = 0.0f;
    ag->forget_surprise_threshold = -1.0f; /* -1 = instantaneous gate, unchanged default */
    ag->forget_locked_off = 0;
    /* RNG seed must be deterministic from the agent's name-seed alone,
     * not from its heap address. Seeding from the pointer (the old
     * behavior) made every process run explore a different random
     * sequence -- and learn different final weights -- purely because
     * heap addresses vary run to run, even with identical training args.
     * That defeated the toolkit's deterministic premise: only weight
     * INIT was reproducible before this fix; training was not. */
    ag->rng_state = 0x9E3779B97F4A7C15ULL;
    for (int rs = 0; rs < 16; rs++) {
        uint64_t chunk; memcpy(&chunk, seed + rs * 8, 8);
        ag->rng_state = zagent__sm64(ag->rng_state ^ chunk);
    }

    memcpy(ag->seed16, seed, 128);

    int actor_in = ag->actor_in;
    int vf_offset = 0;

    /* Hidden layer weights from VirtualField */
    if (hidden_dim) {
        int Nh = hidden_dim * state_dim;
        int Nh_rec = hidden_dim * hidden_dim;
        ag->Wh              = (float*)calloc((size_t)Nh, sizeof(float));
        ag->Wh0             = (float*)calloc((size_t)Nh, sizeof(float));
        ag->bh              = (float*)calloc((size_t)hidden_dim, sizeof(float));
        ag->hidden_act      = (float*)calloc((size_t)hidden_dim, sizeof(float));
        ag->last_hidden_act = (float*)calloc((size_t)hidden_dim, sizeof(float));
        ag->membrane_pot    = (float*)calloc((size_t)hidden_dim, sizeof(float));
        ag->Wh_rec          = (float*)calloc((size_t)Nh_rec, sizeof(float));
        ag->Wh_rec0         = (float*)calloc((size_t)Nh_rec, sizeof(float));
        
        ag->elig_Wh         = (float*)calloc((size_t)Nh, sizeof(float));
        ag->elig_Wh_rec     = (float*)calloc((size_t)Nh_rec, sizeof(float));
        ag->elig_bh         = (float*)calloc((size_t)hidden_dim, sizeof(float));
        ag->pre_h           = (float*)calloc((size_t)state_dim,  sizeof(float));
        
        zagent__vfield(ag->seed16, vf_offset, Nh, ag->Wh);
        memcpy(ag->Wh0, ag->Wh, (size_t)Nh * sizeof(float));
        vf_offset += Nh;
        
        zagent__vfield(ag->seed16, vf_offset, Nh_rec, ag->Wh_rec);
        memcpy(ag->Wh_rec0, ag->Wh_rec, (size_t)Nh_rec * sizeof(float));
        vf_offset += Nh_rec;
    }

    /* Actor layer weights from VirtualField (different region) */
    int Na = n_actions * actor_in;
    ag->Wa      = (float*)calloc((size_t)Na, sizeof(float));
    ag->Wa0     = (float*)calloc((size_t)Na, sizeof(float));
    ag->ba      = (float*)calloc((size_t)n_actions, sizeof(float));
    ag->elig_Wa = (float*)calloc((size_t)Na, sizeof(float));
    ag->elig_ba = (float*)calloc((size_t)n_actions, sizeof(float));
    ag->pre_a   = (float*)calloc((size_t)actor_in,  sizeof(float));
    zagent__vfield(ag->seed16, vf_offset, Na, ag->Wa);
    memcpy(ag->Wa0, ag->Wa, (size_t)Na * sizeof(float));
    vf_offset += Na;

    /* Critic layer weights from VirtualField (next region) */
    int Nc = actor_in;
    ag->Wc        = (float*)calloc((size_t)Nc, sizeof(float));
    ag->Wc0       = (float*)calloc((size_t)Nc, sizeof(float));
    ag->bc        = 0.0f;
    ag->gamma     = 0.96f;
    ag->lr_critic = 0.020f;
    ag->lr_actor  = 0.015f;
    ag->lr_hidden = 0.008f;
    zagent__vfield(ag->seed16, vf_offset, Nc, ag->Wc);
    memcpy(ag->Wc0, ag->Wc, (size_t)Nc * sizeof(float));

    return ag;
}

void zeri_agent_destroy(ZeriAgent* ag) {
    if (!ag) return;
    free(ag->Wh); free(ag->Wh0); free(ag->bh); free(ag->hidden_act);
    free(ag->last_hidden_act); free(ag->membrane_pot);
    free(ag->Wh_rec); free(ag->Wh_rec0);
    free(ag->elig_Wh); free(ag->elig_Wh_rec); free(ag->elig_bh); free(ag->pre_h);
    free(ag->Wa); free(ag->Wa0); free(ag->ba);
    free(ag->elig_Wa); free(ag->elig_ba); free(ag->pre_a);
    free(ag->Wc); free(ag->Wc0);
    free(ag);
}

int zeri_agent_act(ZeriAgent* ag, const float* state, int state_dim) {
    int action = 0;
    /* Epsilon-greedy exploration */
    if (zagent__randf(ag) < ag->epsilon) {
        action = zagent__randi(ag, ag->n_actions);
        // compute forward pass to update SNN membrane potentials and spike history!
        float* lg = (float*)malloc((size_t)ag->n_actions * sizeof(float));
        float* pr = (float*)malloc((size_t)ag->n_actions * sizeof(float));
        zagent__forward(ag, state, lg, pr);
        free(lg); free(pr);
    } else {
        action = zeri_agent_act_greedy(ag, state, state_dim);
    }
    return action;
}

int zeri_agent_act_greedy(ZeriAgent* ag, const float* state, int state_dim) {
    int A = ag->n_actions, D = ag->actor_in;
    float* lg = (float*)malloc((size_t)A * sizeof(float));
    float* pr = (float*)malloc((size_t)A * sizeof(float));
    zagent__forward(ag, state, lg, pr);
    int best = 0;
    for (int a = 1; a < A; a++) if (pr[a] > pr[best]) best = a;
    free(lg); free(pr);
    return best;
}

int zeri_agent_act_stochastic_temp(ZeriAgent* ag, const float* state, int state_dim,
                                    float temperature) {
    int A = ag->n_actions;
    float* lg = (float*)malloc((size_t)A * sizeof(float));
    float* pr = (float*)malloc((size_t)A * sizeof(float));
    zagent__forward(ag, state, lg, pr);

    /* temperature == 1.0 uses the actor's own probabilities as-is (its
     * honest belief, already entropy-regularized). Any other temperature
     * re-derives a distribution from the same logits at that temperature
     * instead of resampling pr[] itself, since pr[] is already a fixed
     * softmax(logits, T=1) and can't be "re-temperatured" after the fact
     * without the original logits. */
    float* sample_pr = pr;
    float* tmp = NULL;
    if (fabsf(temperature - 1.0f) > 1e-6f) {
        float t = temperature > 1e-3f ? temperature : 1e-3f; /* guard div-by-zero */
        tmp = (float*)malloc((size_t)A * sizeof(float));
        float mx = lg[0] / t;
        for (int a = 1; a < A; a++) { float v = lg[a] / t; if (v > mx) mx = v; }
        float s = 0.0f;
        for (int a = 0; a < A; a++) { tmp[a] = expf(lg[a] / t - mx); s += tmp[a]; }
        for (int a = 0; a < A; a++) tmp[a] /= s;
        sample_pr = tmp;
    }

    /* Sample from the (possibly re-temperatured) distribution using the
     * agent's own deterministic RNG stream -- same reproducibility
     * guarantee as epsilon-greedy's random draws. */
    float r = zagent__randf(ag);
    int chosen = A - 1; /* fallback for float rounding: cumulative sum can
                          * land a hair under 1.0 and never trip r < cum */
    float cum = 0.0f;
    for (int a = 0; a < A; a++) {
        cum += sample_pr[a];
        if (r < cum) { chosen = a; break; }
    }

    free(lg); free(pr);
    if (tmp) free(tmp);
    return chosen;
}

int zeri_agent_act_stochastic(ZeriAgent* ag, const float* state, int state_dim) {
    return zeri_agent_act_stochastic_temp(ag, state, state_dim, 1.0f);
}

void zeri_agent_reinforce(ZeriAgent* ag,
                           const float* state, int state_dim,
                           int action,
                           const float* next_state,
                           float reward,
                           int is_terminal) {
    int A = ag->n_actions, D = ag->actor_in;

    /* Compute current state's hidden activations and value */
    const float* actor_in_vec = ag->hidden_dim ? ag->hidden_act : state;
    float v_curr = zagent__dot(ag->Wc, actor_in_vec, D) + ag->bc;

    /* Compute next state's hidden activations and value */
    float v_next = 0.0f;
    if (!is_terminal) {
        float* next_actor_in = (float*)malloc((size_t)D * sizeof(float));
        if (ag->hidden_dim) {
            int H = ag->hidden_dim, S = ag->state_dim;
            for (int h = 0; h < H; h++) {
                float I_input = zagent__dot(ag->Wh + h*S, next_state, S) + ag->bh[h];
                float I_rec = zagent__dot(ag->Wh_rec + h*H, ag->hidden_act, H);
                float next_v = ag->membrane_pot[h] * 0.85f + I_input + I_rec;
                next_actor_in[h] = (next_v >= 1.0f) ? 1.0f : 0.0f;
            }
        } else {
            memcpy(next_actor_in, next_state, (size_t)D * sizeof(float));
        }
        v_next = zagent__dot(ag->Wc, next_actor_in, D) + ag->bc;
        free(next_actor_in);
    }

    /* TD Error: delta = reward + gamma * v_next - v_curr */
    float td_error = reward + ag->gamma * v_next - v_curr;

    /* Rolling EMA of |td_error| -- "how surprised has this agent been
     * lately," as opposed to the instantaneous value used by the default
     * forgetting gate below. ~100-step time constant (0.99 decay). Always
     * maintained (one multiply-add) but only read if
     * forget_surprise_threshold >= 0 -- otherwise has zero effect. */
    ag->avg_abs_td = ag->avg_abs_td * 0.99f + fabsf(td_error) * 0.01f;

    /* Update Critic: Wc += lr_critic * td_error * actor_in */
    for (int d = 0; d < D; d++) {
        ag->Wc[d] += ag->lr_critic * td_error * actor_in_vec[d];
        ag->Wc[d]  = zagent__clamp(ag->Wc[d], -ZAGENT_CLIP, ZAGENT_CLIP);
    }
    ag->bc += ag->lr_critic * td_error;

    /* For Actor update: forward pass to get current probs */
    float* lg = (float*)malloc((size_t)A * sizeof(float));
    float* pr = (float*)malloc((size_t)A * sizeof(float));
    zagent__actor_fwd(ag, actor_in_vec, lg, pr);

    /* Update eligibility traces using current state and chosen action */
    zagent__update_eligibility(ag, state, actor_in_vec, pr, action);

    /* Entropy regularization: nudge the policy toward higher entropy by a
     * small, reward-INDEPENDENT amount every step (unlike the eligibility
     * traces above, which get scaled by td_error/reward in
     * zagent__apply_reward -- entropy must not flip sign with the reward,
     * or a negative reward would actively sharpen the saturation instead
     * of relieving it).
     *
     * Why this exists: vanilla REINFORCE/actor-critic has a well-known
     * failure mode where a heavily and consistently reinforced action
     * pushes its softmax probability to ~1.0. Once saturated, the policy
     * gradient term (1_chosen - prob) is ~0 for every action, so the
     * gradient that would normally correct a wrong choice in a *different*
     * state (here: DOWN at the post-key cell of the Vault preset, after
     * DOWN was the correct, heavily-reinforced move for the 6 steps
     * leading up to it) vanishes -- no amount of subsequent negative
     * reward can move the policy back off it. Entropy regularization
     * keeps a small gradient flowing even near saturation, so learning
     * doesn't permanently stall. Standard technique (A2C/PPO); applied
     * here directly to the linear actor layer. */
    {
        float H = 0.0f;
        for (int a = 0; a < A; a++) {
            float p = pr[a] > 1e-6f ? pr[a] : 1e-6f;
            H -= p * logf(p);
        }
        float ent_lr = ag->lr_actor * ZAGENT_ENTROPY_BETA;
        for (int a = 0; a < A; a++) {
            float p = pr[a] > 1e-6f ? pr[a] : 1e-6f;
            float g_ent = -p * (logf(p) + H);
            ag->ba[a] += ent_lr * g_ent;
            ag->ba[a]  = zagent__clamp(ag->ba[a], -ZAGENT_CLIP, ZAGENT_CLIP);
            float* Wa_row = ag->Wa + a * D;
            for (int d = 0; d < D; d++) {
                Wa_row[d] += ent_lr * g_ent * actor_in_vec[d];
                Wa_row[d]  = zagent__clamp(Wa_row[d], -ZAGENT_CLIP, ZAGENT_CLIP);
            }
        }
    }

    free(lg); free(pr);

    /* Decay pre-synaptic traces, then accumulate current input */
    for (int d = 0; d < D; d++) ag->pre_a[d] = ag->pre_a[d] * 0.9f + actor_in_vec[d];
    if (ag->hidden_dim) {
        for (int s = 0; s < ag->state_dim; s++) {
            ag->pre_h[s] = ag->pre_h[s] * 0.9f + state[s];
        }
    }

    /* Update Actor using td_error instead of raw reward */
    zagent__apply_reward(ag, td_error);

    /* Biological Synaptic Forgetting (L1 delta decay) - coupled to TD error.
     * Default gate is instantaneous (fabsf(td_error) > 1e-4f) -- kept
     * exactly as originally written for backward compatibility, even
     * though it's nearly always true for a continuous float and so isn't
     * really selective in practice. zeri_agent_set_forget_surprise_gate
     * opts into a rolling-average gate with a one-way latch instead: see
     * that function's doc comment for why the latch specifically was
     * needed (a non-latching toggle was tried first and was unreliable). */
    if (ag->forget_surprise_threshold >= 0.0f && !ag->forget_locked_off
        && ag->avg_abs_td < ag->forget_surprise_threshold) {
        ag->forget_locked_off = 1;
    }
    int zagent__should_forget = (ag->forget_surprise_threshold >= 0.0f)
        ? (!ag->forget_locked_off && ag->avg_abs_td > ag->forget_surprise_threshold)
        : (fabsf(td_error) > 1e-4f);
    if (zagent__should_forget && ag->forget_rate > 0.0f) {
        float f_rate = ag->forget_rate;
        for (int i = 0; i < A * D; i++) {
            float d = ag->Wa[i] - ag->Wa0[i];
            if      (d >  f_rate) d -= f_rate;
            else if (d < -f_rate) d += f_rate;
            else                  d = 0.0f;
            ag->Wa[i] = ag->Wa0[i] + d;
        }
        if (ag->hidden_dim) {
            int H = ag->hidden_dim, S = ag->state_dim;
            for (int i = 0; i < H * S; i++) {
                float d = ag->Wh[i] - ag->Wh0[i];
                if      (d >  f_rate) d -= f_rate;
                else if (d < -f_rate) d += f_rate;
                else                  d = 0.0f;
                ag->Wh[i] = ag->Wh0[i] + d;
            }
            for (int i = 0; i < H * H; i++) {
                float d = ag->Wh_rec[i] - ag->Wh_rec0[i];
                if      (d >  f_rate) d -= f_rate;
                else if (d < -f_rate) d += f_rate;
                else                  d = 0.0f;
                ag->Wh_rec[i] = ag->Wh_rec0[i] + d;
            }
        }
        for (int i = 0; i < D; i++) {
            float d = ag->Wc[i] - ag->Wc0[i];
            if      (d >  f_rate) d -= f_rate;
            else if (d < -f_rate) d += f_rate;
            else                  d = 0.0f;
            ag->Wc[i] = ag->Wc0[i] + d;
        }
    }

    ag->steps++;
    ag->total_reward += (double)reward;
    ag->reward_window[ag->reward_win_idx++ % 100] = reward;
}

void zeri_agent_reset_traces(ZeriAgent* ag) {
    if (!ag) return;
    int A = ag->n_actions, D = ag->actor_in;
    memset(ag->elig_Wa, 0, (size_t)A * D * sizeof(float));
    memset(ag->elig_ba, 0, (size_t)A * sizeof(float));
    memset(ag->pre_a,   0, (size_t)D * sizeof(float));
    if (ag->hidden_dim) {
        int H = ag->hidden_dim, S = ag->state_dim;
        memset(ag->elig_Wh, 0, (size_t)H * S * sizeof(float));
        memset(ag->elig_bh, 0, (size_t)H * sizeof(float));
        memset(ag->elig_Wh_rec, 0, (size_t)H * H * sizeof(float));
        memset(ag->pre_h,   0, (size_t)S * sizeof(float));
        memset(ag->membrane_pot, 0, (size_t)H * sizeof(float));
        memset(ag->hidden_act, 0, (size_t)H * sizeof(float));
        memset(ag->last_hidden_act, 0, (size_t)H * sizeof(float));
    }
}

void zeri_agent_decay_epsilon(ZeriAgent* ag) {
    ag->episodes++;
    ag->epsilon *= ag->eps_decay;
    if (ag->epsilon < ag->eps_min) ag->epsilon = ag->eps_min;

    /* Dynamic developmental learning rate crystallization */
    ag->lr_actor *= 0.998f;
    if (ag->lr_actor < 0.002f) ag->lr_actor = 0.002f;
    ag->lr_hidden *= 0.998f;
    if (ag->lr_hidden < 0.001f) ag->lr_hidden = 0.001f;
    ag->lr_critic *= 0.998f;
    if (ag->lr_critic < 0.003f) ag->lr_critic = 0.003f;
}

void zeri_agent_set_decay(ZeriAgent* ag, float rate, float min_eps) {
    ag->eps_decay = rate;
    ag->eps_min   = min_eps;
}

void zeri_agent_set_forget_rate(ZeriAgent* ag, float rate) {
    ag->forget_rate = rate;
}

void zeri_agent_set_forget_surprise_gate(ZeriAgent* ag, float threshold) {
    ag->forget_surprise_threshold = threshold;
    ag->forget_locked_off = 0; /* re-arm: a fresh threshold gets a fresh chance to earn its own lock */
}

float zeri_agent_get_avg_abs_td(ZeriAgent* ag) {
    return ag->avg_abs_td;
}

void zeri_agent_probs(ZeriAgent* ag, const float* state, int state_dim,
                      float* out_probs) {
    int A = ag->n_actions;
    float* lg = (float*)malloc((size_t)A * sizeof(float));
    zagent__forward(ag, state, lg, out_probs);
    free(lg);
}

void zeri_agent_stats(ZeriAgent* ag, float* out) {
    out[0] = ag->epsilon;
    out[1] = (float)ag->episodes;
    out[2] = (float)ag->total_reward;
    /* average of last 100 rewards */
    int n = ag->steps < 100 ? (int)ag->steps : 100;
    float avg = 0.0f;
    for (int i = 0; i < n; i++) avg += ag->reward_window[i];
    out[3] = n > 0 ? avg / n : 0.0f;
    out[4] = (float)ag->steps;
}

static inline void zagent__write_sparse(const float* w, const float* w0, int count, float thresh, uint8_t* buf, int* off) {
    uint32_t c = 0;
    for (int i = 0; i < count; i++) {
        if (fabsf(w[i] - w0[i]) >= thresh) c++;
    }
    memcpy(buf + *off, &c, 4); *off += 4;
    for (int i = 0; i < count; i++) {
        float d = w[i] - w0[i];
        if (fabsf(d) >= thresh) {
            uint32_t idx = (uint32_t)i;
            memcpy(buf + *off, &idx, 4); *off += 4;
            memcpy(buf + *off, &d, 4); *off += 4;
        }
    }
}

static inline void zagent__read_sparse(float* w, const float* w0, int count, const uint8_t* buf, int* off) {
    memcpy(w, w0, (size_t)count * sizeof(float));
    uint32_t c; memcpy(&c, buf + *off, 4); *off += 4;
    for (uint32_t j = 0; j < c; j++) {
        uint32_t idx; memcpy(&idx, buf + *off, 4); *off += 4;
        float d; memcpy(&d, buf + *off, 4); *off += 4;
        if (idx < (uint32_t)count) {
            w[idx] = w0[idx] + d;
        }
    }
}

/* v2: lossless, smaller. Indices are written in ascending scan order, so each
 * one is encoded as a GAP from the previous index instead of a flat 4-byte
 * absolute index -- 1 byte for gaps < 255, a 5-byte escape (0xFF + uint32)
 * otherwise. Delta values are untouched (still exact float32). Measured on a
 * real 20k-step training run: ~1.6-1.8x smaller than v1, byte-for-byte exact
 * round trip (see mosaic/memory_format_results.txt). v1 read/write above are
 * kept as-is so existing checkpoint files keep loading unmodified. */
static inline void zagent__write_sparse_v2(const float* w, const float* w0, int count, float thresh, uint8_t* buf, int* off) {
    uint32_t c = 0;
    for (int i = 0; i < count; i++) {
        if (fabsf(w[i] - w0[i]) >= thresh) c++;
    }
    memcpy(buf + *off, &c, 4); *off += 4;
    int32_t prev = -1;
    for (int i = 0; i < count; i++) {
        float d = w[i] - w0[i];
        if (fabsf(d) >= thresh) {
            uint32_t gap = (uint32_t)(i - prev); /* always >= 1: i is strictly increasing */
            if (gap < 255) {
                uint8_t g8 = (uint8_t)gap;
                memcpy(buf + *off, &g8, 1); *off += 1;
            } else {
                uint8_t esc = 255;
                memcpy(buf + *off, &esc, 1); *off += 1;
                memcpy(buf + *off, &gap, 4); *off += 4;
            }
            memcpy(buf + *off, &d, 4); *off += 4;
            prev = i;
        }
    }
}

static inline void zagent__read_sparse_v2(float* w, const float* w0, int count, const uint8_t* buf, int* off) {
    memcpy(w, w0, (size_t)count * sizeof(float));
    uint32_t c; memcpy(&c, buf + *off, 4); *off += 4;
    int32_t prev = -1;
    for (uint32_t j = 0; j < c; j++) {
        uint8_t g8; memcpy(&g8, buf + *off, 1); *off += 1;
        uint32_t gap;
        if (g8 < 255) {
            gap = g8;
        } else {
            memcpy(&gap, buf + *off, 4); *off += 4;
        }
        int32_t idx = prev + (int32_t)gap;
        prev = idx;
        float d; memcpy(&d, buf + *off, 4); *off += 4;
        if (idx >= 0 && idx < count) {
            w[idx] = w0[idx] + d;
        }
    }
}

int zeri_agent_consolidate(ZeriAgent* ag, uint8_t* buf, int buf_size) {
    int Na = ag->n_actions * ag->actor_in;
    int Nh = ag->hidden_dim * ag->state_dim;
    int Nh_rec = ag->hidden_dim * ag->hidden_dim;
    int Nc = ag->actor_in;
    
    uint32_t magic = 0x32505A45; /* "ZSP2" - gap-encoded indices, same exact float32 deltas as v1 */
    memcpy(buf,      &magic,           4);
    memcpy(buf +  4, &ag->state_dim,   4);
    memcpy(buf +  8, &ag->n_actions,   4);
    memcpy(buf + 12, &ag->hidden_dim,  4);
    int off = 16;

    zagent__write_sparse_v2(ag->Wa, ag->Wa0, Na, 1e-6f, buf, &off);
    memcpy(buf + off, ag->ba, (size_t)ag->n_actions * sizeof(float)); off += ag->n_actions * 4;

    if (ag->hidden_dim) {
        zagent__write_sparse_v2(ag->Wh, ag->Wh0, Nh, 1e-6f, buf, &off);
        zagent__write_sparse_v2(ag->Wh_rec, ag->Wh_rec0, Nh_rec, 1e-6f, buf, &off);
        memcpy(buf + off, ag->bh, (size_t)ag->hidden_dim * sizeof(float)); off += ag->hidden_dim * 4;
    }

    zagent__write_sparse_v2(ag->Wc, ag->Wc0, Nc, 1e-6f, buf, &off);
    memcpy(buf + off, &ag->bc, 4); off += 4;

    return off;
}

int zeri_agent_load_memory(ZeriAgent* ag, const uint8_t* buf, int buf_size) {
    if (buf_size < 16) return -1;
    uint32_t magic; memcpy(&magic, buf, 4);
    int is_v2;
    if (magic == 0x32505A45) {       /* "ZSP2" - current format, written by this build */
        is_v2 = 1;
    } else if (magic == 0x5A455350) { /* "ZESP" - old format, kept readable for existing checkpoints */
        is_v2 = 0;
    } else {
        return -2;
    }

    int off = 16;
    int Na = ag->n_actions * ag->actor_in;
    int Nh = ag->hidden_dim * ag->state_dim;
    int Nh_rec = ag->hidden_dim * ag->hidden_dim;
    int Nc = ag->actor_in;

    if (is_v2) zagent__read_sparse_v2(ag->Wa, ag->Wa0, Na, buf, &off);
    else       zagent__read_sparse(ag->Wa, ag->Wa0, Na, buf, &off);
    memcpy(ag->ba, buf + off, (size_t)ag->n_actions * sizeof(float)); off += ag->n_actions * 4;

    if (ag->hidden_dim) {
        if (is_v2) {
            zagent__read_sparse_v2(ag->Wh, ag->Wh0, Nh, buf, &off);
            zagent__read_sparse_v2(ag->Wh_rec, ag->Wh_rec0, Nh_rec, buf, &off);
        } else {
            zagent__read_sparse(ag->Wh, ag->Wh0, Nh, buf, &off);
            zagent__read_sparse(ag->Wh_rec, ag->Wh_rec0, Nh_rec, buf, &off);
        }
        memcpy(ag->bh, buf + off, (size_t)ag->hidden_dim * sizeof(float)); off += ag->hidden_dim * 4;
    }

    if (is_v2) zagent__read_sparse_v2(ag->Wc, ag->Wc0, Nc, buf, &off);
    else       zagent__read_sparse(ag->Wc, ag->Wc0, Nc, buf, &off);
    memcpy(&ag->bc, buf + off, 4); off += 4;

    return off;
}

ZeriAgent* zeri_agent_from_name(const char* name,
                                  int state_dim, int n_actions, int hidden_dim,
                                  float epsilon) {
    uint8_t seed[128] = {0};
    uint64_t h = 14695981039346656037ULL;
    for (const char* p = name; *p; p++) { h ^= (uint8_t)*p; h *= 1099511628211ULL; }
    for (int i = 0; i < 16; i++) { h = zagent__sm64(h+(uint64_t)i); memcpy(seed+i*8,&h,8); }
    return zeri_agent_create(seed, state_dim, n_actions, hidden_dim, epsilon);
}

/* =========================================================================
   GENETIC / BREEDING IMPLEMENTATIONS
   ========================================================================= */

ZeriAgent* zeri_agent_breed_seed(const ZeriAgent* pa, const ZeriAgent* pb,
                                  const uint8_t* offspring_seed,
                                  float alpha, float epsilon) {
    /* Validate compatible architectures */
    if (pa->state_dim != pb->state_dim || pa->n_actions != pb->n_actions ||
        pa->hidden_dim != pb->hidden_dim) return NULL;

    /* Create offspring with its own personality */
    ZeriAgent* child = zeri_agent_create(offspring_seed,
                                          pa->state_dim, pa->n_actions,
                                          pa->hidden_dim, epsilon);
    if (!child) return NULL;

    float beta = 1.0f - alpha;
    int Na = pa->n_actions * pa->actor_in;
    int Nc = pa->actor_in;

    /* Blend actor weight deltas onto child's own baseline */
    for (int i = 0; i < Na; i++) {
        float da = pa->Wa[i] - pa->Wa0[i];
        float db = pb->Wa[i] - pb->Wa0[i];
        child->Wa[i] = zagent__clamp(child->Wa0[i] + alpha*da + beta*db,
                                      -ZAGENT_CLIP, ZAGENT_CLIP);
    }
    for (int i = 0; i < pa->n_actions; i++) {
        child->ba[i] = zagent__clamp(alpha*pa->ba[i] + beta*pb->ba[i],
                                      -ZAGENT_CLIP, ZAGENT_CLIP);
    }
    /* Blend critic */
    for (int i = 0; i < Nc; i++) {
        float da = pa->Wc[i] - pa->Wc0[i];
        float db = pb->Wc[i] - pb->Wc0[i];
        child->Wc[i] = zagent__clamp(child->Wc0[i] + alpha*da + beta*db,
                                      -ZAGENT_CLIP, ZAGENT_CLIP);
    }
    child->bc = alpha*pa->bc + beta*pb->bc;

    /* Blend hidden layer if present */
    if (pa->hidden_dim) {
        int H = pa->hidden_dim, S = pa->state_dim;
        int Nh = H * S, Nh_rec = H * H;
        for (int i = 0; i < Nh; i++) {
            float da = pa->Wh[i] - pa->Wh0[i];
            float db = pb->Wh[i] - pb->Wh0[i];
            child->Wh[i] = zagent__clamp(child->Wh0[i] + alpha*da + beta*db,
                                          -ZAGENT_CLIP, ZAGENT_CLIP);
        }
        for (int i = 0; i < Nh_rec; i++) {
            float da = pa->Wh_rec[i] - pa->Wh_rec0[i];
            float db = pb->Wh_rec[i] - pb->Wh_rec0[i];
            child->Wh_rec[i] = zagent__clamp(child->Wh_rec0[i] + alpha*da + beta*db,
                                              -ZAGENT_CLIP, ZAGENT_CLIP);
        }
        for (int i = 0; i < H; i++) {
            child->bh[i] = zagent__clamp(alpha*pa->bh[i] + beta*pb->bh[i],
                                          -ZAGENT_CLIP, ZAGENT_CLIP);
        }
    }
    return child;
}

ZeriAgent* zeri_agent_breed(const ZeriAgent* pa, const ZeriAgent* pb,
                             const char* offspring_name,
                             float alpha, float epsilon) {
    uint8_t seed[128] = {0};
    uint64_t h = 14695981039346656037ULL;
    for (const char* p = offspring_name; *p; p++) { h ^= (uint8_t)*p; h *= 1099511628211ULL; }
    for (int i = 0; i < 16; i++) { h = zagent__sm64(h+(uint64_t)i); memcpy(seed+i*8,&h,8); }
    return zeri_agent_breed_seed(pa, pb, seed, alpha, epsilon);
}

ZeriAgent* zeri_agent_clone(const ZeriAgent* src) {
    uint8_t seed[128];
    memcpy(seed, src->seed16, 128);
    ZeriAgent* dst = zeri_agent_create(seed, src->state_dim, src->n_actions,
                                        src->hidden_dim, src->epsilon);
    if (!dst) return NULL;

    int Na = src->n_actions * src->actor_in;
    int Nc = src->actor_in;
    memcpy(dst->Wa, src->Wa, (size_t)Na * sizeof(float));
    memcpy(dst->ba, src->ba, (size_t)src->n_actions * sizeof(float));
    memcpy(dst->Wc, src->Wc, (size_t)Nc * sizeof(float));
    dst->bc = src->bc;
    dst->epsilon = src->epsilon;
    dst->eps_decay = src->eps_decay;
    dst->eps_min = src->eps_min;
    dst->steps = src->steps;
    dst->episodes = src->episodes;
    dst->total_reward = src->total_reward;
    dst->reward_win_idx = src->reward_win_idx;
    memcpy(dst->reward_window, src->reward_window, sizeof(src->reward_window));
    if (src->hidden_dim) {
        int H = src->hidden_dim, S = src->state_dim;
        memcpy(dst->Wh,     src->Wh,     (size_t)H*S * sizeof(float));
        memcpy(dst->Wh_rec, src->Wh_rec, (size_t)H*H * sizeof(float));
        memcpy(dst->bh,     src->bh,     (size_t)H   * sizeof(float));
        memcpy(dst->hidden_act,      src->hidden_act,      (size_t)H * sizeof(float));
        memcpy(dst->membrane_pot,    src->membrane_pot,    (size_t)H * sizeof(float));
        memcpy(dst->last_hidden_act, src->last_hidden_act, (size_t)H * sizeof(float));
    }
    return dst;
}

float zeri_agent_fitness(ZeriAgent* ag, int max_steps) {
    /* avg_reward_100 is already in the reward window */
    int n = ag->steps < 100 ? (int)ag->steps : 100;
    if (n == 0) return 0.0f;
    float avg = 0.0f;
    for (int i = 0; i < n; i++) avg += ag->reward_window[i];
    avg /= (float)n;
    /* Normalize: best possible is +10/episode of ~12 steps = ~0.83/step; worst ≈ -0.5 */
    return avg;
}

float zeri_agent_get_delta_norm(const ZeriAgent* ag) {
    float norm = 0.0f;
    int Na = ag->n_actions * ag->actor_in;
    for (int i = 0; i < Na; i++) norm += fabsf(ag->Wa[i] - ag->Wa0[i]);
    for (int i = 0; i < ag->n_actions; i++) norm += fabsf(ag->ba[i]);
    if (ag->hidden_dim) {
        int H = ag->hidden_dim, S = ag->state_dim;
        for (int i = 0; i < H*S; i++) norm += fabsf(ag->Wh[i] - ag->Wh0[i]);
        for (int i = 0; i < H*H; i++) norm += fabsf(ag->Wh_rec[i] - ag->Wh_rec0[i]);
    }
    for (int i = 0; i < ag->actor_in; i++) norm += fabsf(ag->Wc[i] - ag->Wc0[i]);
    return norm;
}

void zeri_agent_mutate(ZeriAgent* ag, float sigma) {
    /* Simple Box-Muller Gaussian noise via the agent's own RNG */
    int Na = ag->n_actions * ag->actor_in;
    for (int i = 0; i < Na; i++) {
        float u1 = zagent__randf(ag) + 1e-9f;
        float u2 = zagent__randf(ag);
        float gauss = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        ag->Wa[i] = zagent__clamp(ag->Wa[i] + sigma * gauss, -ZAGENT_CLIP, ZAGENT_CLIP);
    }
    if (ag->hidden_dim) {
        int H = ag->hidden_dim, S = ag->state_dim;
        for (int i = 0; i < H*S; i++) {
            float u1 = zagent__randf(ag) + 1e-9f;
            float u2 = zagent__randf(ag);
            float gauss = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
            ag->Wh[i] = zagent__clamp(ag->Wh[i] + sigma * gauss, -ZAGENT_CLIP, ZAGENT_CLIP);
        }
    }
}

float zeri_agent_value(ZeriAgent* ag, const float* state, int state_dim) {
    zagent__hidden_fwd(ag, state);
    const float* actor_in = ag->hidden_dim ? ag->hidden_act : state;
    return zagent__dot(ag->Wc, actor_in, ag->actor_in) + ag->bc;
}

float zeri_agent_policy_entropy(ZeriAgent* ag, const float* state, int state_dim) {
    int A = ag->n_actions;
    float* lg = (float*)malloc((size_t)A * sizeof(float));
    float* pr = (float*)malloc((size_t)A * sizeof(float));
    zagent__forward(ag, state, lg, pr);
    float H = 0.0f;
    for (int a = 0; a < A; a++) {
        float p = pr[a] > 1e-9f ? pr[a] : 1e-9f;
        H -= p * logf(p);
    }
    free(lg); free(pr);
    return H;
}

void zeri_agent_reset_stats(ZeriAgent* ag) {
    ag->steps = 0;
    ag->episodes = 0;
    ag->total_reward = 0.0;
    ag->reward_win_idx = 0;
    memset(ag->reward_window, 0, sizeof(ag->reward_window));
}

#endif /* ZERI_AGENT_IMPLEMENTATION */
#endif /* ZERI_AGENT_H */
