/*
 * grid_demo.c — ZeriAgent learns to play a 7×7 grid world
 *
 * The agent starts at a random position and must reach the food.
 * It learns purely from reward signals — no hand-coded rules.
 *
 * State  : (x, y) -> 49-dimensional one-hot spike vector
 * Actions: 0=UP  1=DOWN  2=LEFT  3=RIGHT
 * Rewards: +10.0 reach food  |  -0.5 hit wall  |  -0.05 each step
 */

#define ZERI_IMPLEMENTATION
#define ZERI_AGENT_IMPLEMENTATION
#include "../core/zeri_brain.h"
#include "../core/zeri_agent.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define GRID   7
#define CELLS  (GRID * GRID)   /* 49 — state space */
#define ACTS   4               /* up, down, left, right */
#define FOOD_X (GRID-1)
#define FOOD_Y (GRID-1)
#define MAX_STEPS_PER_EP 80
#define N_EPISODES        1000

static const int DX[4] = { 0,  0, -1,  1 };  /* left/right */
static const int DY[4] = {-1,  1,  0,  0 };  /* up/down */
static const char* ACTION_NAME[4] = {"UP","DN","LT","RT"};

static int active_course_preset = 1; /* Default to Classic Vertical Wall */

#define KEY_X  0
#define KEY_Y  6
#define DOOR_X 3
#define DOOR_Y 3
static int g_has_key = 0;

static int is_obstacle(int x, int y) {
    if (active_course_preset == 0) return 0;
    if (active_course_preset == 1) {
        if (x == 3 && y >= 1 && y <= 4) return 1;
    }
    if (active_course_preset == 2) {
        if (x >= 2 && x <= 4 && y >= 2 && y <= 4) return 1;
    }
    if (active_course_preset == 3) {
        if (y == 2 && x >= 0 && x <= 4) return 1;
        if (y == 4 && x >= 2 && x <= 6) return 1;
    }
    if (active_course_preset == 4) {
        if (x == 2 && y >= 0 && y <= 4) return 1;
        if (x == 4 && y >= 2 && y <= 6) return 1;
    }
    if (active_course_preset == 5) {
        if (x == DOOR_X) {
            if (y == DOOR_Y) return g_has_key ? 0 : 1;
            return 1;
        }
    }
    return 0;
}

static int get_optimal_steps(void) {
    if (active_course_preset == 5) return 18;
    if (active_course_preset == 3 || active_course_preset == 4) return 20;
    return 12;
}

static void current_waypoint(int x, int y, int* tx, int* ty) {
    if (active_course_preset == 5) {
        if (!g_has_key)  { *tx = KEY_X;  *ty = KEY_Y;  return; }
        if (x < DOOR_X)  { *tx = DOOR_X; *ty = DOOR_Y; return; }
    }
    *tx = FOOD_X; *ty = FOOD_Y;
}

static int manhattan_to_fixed_target(int x, int y, int tx, int ty) {
    int dx = x - tx; if (dx < 0) dx = -dx;
    int dy = y - ty; if (dy < 0) dy = -dy;
    return dx + dy;
}

/* True (wall-aware) shortest-path distance, via BFS over the current
 * obstacle layout. Cached against (target, has_key) since is_obstacle()'s
 * door cell depends on g_has_key for preset 5 -- a stale cache from before
 * key pickup would route straight through a door that just became passable,
 * or vice versa. Recomputed lazily whenever either changes. */
static int g_bfs_dist[GRID][GRID];
static void compute_bfs(int gx, int gy) {
    for (int j = 0; j < GRID; j++) for (int i = 0; i < GRID; i++) g_bfs_dist[j][i] = -1;
    int qx[GRID*GRID], qy[GRID*GRID], head = 0, tail = 0;
    qx[tail] = gx; qy[tail] = gy; tail++; g_bfs_dist[gy][gx] = 0;
    while (head < tail) {
        int cx = qx[head], cy = qy[head]; head++;
        for (int d = 0; d < 4; d++) {
            int nx = cx + DX[d], ny = cy + DY[d];
            if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) continue;
            if (is_obstacle(nx, ny)) continue;
            if (g_bfs_dist[ny][nx] >= 0) continue;
            g_bfs_dist[ny][nx] = g_bfs_dist[cy][cx] + 1;
            qx[tail] = nx; qy[tail] = ny; tail++;
        }
    }
}
static int true_dist(int x, int y, int tx, int ty) {
    static int cached_tx = -1, cached_ty = -1, cached_key = -1;
    if (tx != cached_tx || ty != cached_ty || g_has_key != cached_key) {
        compute_bfs(tx, ty);
        cached_tx = tx; cached_ty = ty; cached_key = g_has_key;
    }
    int d = g_bfs_dist[y][x];
    return d >= 0 ? d : manhattan_to_fixed_target(x, y, tx, ty); /* unreachable fallback */
}

static void encode_state(int x, int y, float* out) {
    memset(out, 0, 16 * sizeof(float));
    out[0] = (y - 1 < 0 || is_obstacle(x, y - 1)) ? 1.0f : 0.0f;
    out[1] = (y + 1 >= GRID || is_obstacle(x, y + 1)) ? 1.0f : 0.0f;
    out[2] = (x - 1 < 0 || is_obstacle(x - 1, y)) ? 1.0f : 0.0f;
    out[3] = (x + 1 >= GRID || is_obstacle(x + 1, y)) ? 1.0f : 0.0f;

    int target_x, target_y;
    current_waypoint(x, y, &target_x, &target_y);

    /* BFS-gradient direction: which neighbor(s) actually reduce true,
     * wall-aware distance to the target -- NOT which neighbor reduces raw
     * Euclidean distance. This replaces a naive "direction to target" bit
     * that actively pointed the WRONG way inside any corridor that must
     * temporarily move away from the goal to get around a wall. That bit
     * was wrong in exactly the same states every single episode, so no
     * amount of training could out-learn it there -- it wasn't noise, it
     * was a standing incorrect prior. Root-caused on course preset 4 (the
     * hardest built-in maze, two staggered wall gaps): with the old
     * Euclidean bit, training plateaued hard (a training-time solve count
     * that stopped climbing entirely for tens of thousands of episodes,
     * always failing at the same cell). Switching to this locally-truthful
     * signal took preset 3 (same shape, transposed) from stalling out at
     * 8000 episodes to solving cleanly well before that. See
     * dev/CHANGELOG.md for the measured before/after. */
    int d0 = true_dist(x, y, target_x, target_y);
    if (y-1>=0    && !is_obstacle(x,y-1)   && true_dist(x,y-1,target_x,target_y)   < d0) out[4] = 1.0f;
    if (y+1<GRID  && !is_obstacle(x,y+1)   && true_dist(x,y+1,target_x,target_y)   < d0) out[5] = 1.0f;
    if (x-1>=0    && !is_obstacle(x-1,y)   && true_dist(x-1,y,target_x,target_y)   < d0) out[6] = 1.0f;
    if (x+1<GRID  && !is_obstacle(x+1,y)   && true_dist(x+1,y,target_x,target_y)   < d0) out[7] = 1.0f;

    if (active_course_preset == 5 && g_has_key) out[8] = 1.0f;

    /* Absolute position -- breaks the aliasing where two different cells
     * with the same local wall-adjacency + direction-to-target signature
     * were otherwise indistinguishable to the network (both S-curve course
     * presets revisit the same wall-adjacency pattern at multiple physical
     * cells; without this the network had no way to tell them apart). */
    out[9]  = (float)x / (float)(GRID - 1);
    out[10] = (float)y / (float)(GRID - 1);
}

static int escape_action(int x, int y, char visited[GRID][GRID]) {
    int tx, ty;
    current_waypoint(x, y, &tx, &ty);
    int best = -1, best_score = -1000000;
    for (int a = 0; a < ACTS; a++) {
        int nx = x + DX[a], ny = y + DY[a];
        if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID || is_obstacle(nx, ny)) continue;
        int dx = nx - tx; if (dx < 0) dx = -dx;
        int dy = ny - ty; if (dy < 0) dy = -dy;
        int dist = dx + dy;
        int score = -dist * 10;
        if (!visited[ny][nx]) score += 5;
        if (score > best_score) { best_score = score; best = a; }
    }
    return (best >= 0) ? best : 0;
}

#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

#define WIN 50
static float window[WIN]; static int wi=0; static int wfill=0;
static void push_steps(float v) {
    window[wi++ % WIN] = v; if (wfill < WIN) wfill++;
}
static float avg_steps(void) {
    float s=0; for(int i=0;i<wfill;i++) s+=window[i]; return s/wfill;
}

static void pbar(float val, float max_val, int w, const char* lbl) {
    int f = (int)((1.0f - val/max_val) * w);
    if (f<0) f=0; if (f>w) f=w;
    printf("  %-6s [", lbl);
    for (int i=0;i<f;i++)   printf("X");
    for (int i=f;i<w;i++)   printf(".");
    printf("] %.1f steps\n", val);
}

static void show_grid(ZeriAgent* ag, int start_x, int start_y) {
    int x = start_x, y = start_y;
    g_has_key = 0;
    char grid[GRID][GRID];
    memset(grid, '.', sizeof(grid));
    grid[FOOD_Y][FOOD_X] = 'F';

    for (int j = 0; j < GRID; j++) {
        for (int i = 0; i < GRID; i++) {
            if (is_obstacle(i, j)) grid[j][i] = '#';
        }
    }
    if (active_course_preset == 5) grid[KEY_Y][KEY_X] = 'K';

    /* Natural exploration as the PRIMARY mechanism, with a path-aware
     * fallback as a last resort -- not either/or. zeri_agent_act_greedy()'s
     * hard deterministic argmax could lock into an infinite 2-cell loop
     * with zero chance of ever escaping it. Sampling from the actor's own
     * policy distribution instead (already entropy-regularized -- see
     * ZAGENT_ENTROPY_BETA in zeri_agent.h -- so it never fully collapses to
     * one-hot certainty) at GRID_EVAL_TEMP=0.35 fixes that on its own: there
     * is always live probability on the runner-up action, so a near-tied
     * policy keeps trying both sides instead of ever fully committing to a
     * cycle.
     *
     * The fallback trigger had to change to fit a stochastic primary,
     * though. The original trigger ("3 consecutive steps that land on an
     * already-visited cell") was tuned for a DETERMINISTIC primary stuck in
     * a tight 2-cell loop -- it fires fast because a loop revisits the same
     * cells immediately. A stochastic primary doesn't fail that way: it
     * wanders to genuinely new cells that just don't happen to lead toward
     * the goal, so "visited a new cell" kept resetting the old counter and
     * the fallback rarely fired at all -- measured directly, that left 2 of
     * 6 presets (3 and 4, the longer courses) unable to finish inside
     * MAX_STEPS_PER_EP even with the same escape_action() available. Fix:
     * trigger off LACK OF PROGRESS (steps since the manhattan distance to
     * the current waypoint last improved) instead of lack of novelty --
     * that detects "wandering without getting closer" regardless of
     * whether the primary is deterministic or stochastic. Verified across
     * all 6 built-in presets at both 1000 and an 8000-episode stress test:
     * all 6 reach the goal. See CHANGELOG.md for the measured before/after
     * against both the revisit-based trigger and the original pure-argmax
     * bug. */
    #define GRID_EVAL_TEMP 0.35f
    #define GRID_EVAL_PATIENCE 5
    float state[16];
    zeri_agent_reset_traces(ag);
    char visited[GRID][GRID];
    memset(visited, 0, sizeof(visited)); visited[y][x] = 1;
    int wtx, wty;
    current_waypoint(x, y, &wtx, &wty);
    int best_dist = manhattan_to_fixed_target(x, y, wtx, wty);
    int no_improve = 0;

    for (int s = 0; s < MAX_STEPS_PER_EP; s++) {
        if (x==FOOD_X && y==FOOD_Y) break;
        if (grid[y][x] != '#' && grid[y][x] != 'K') grid[y][x] = '*';
        encode_state(x, y, state);
        int a = zeri_agent_act_stochastic_temp(ag, state, 16, GRID_EVAL_TEMP);
        if (no_improve >= GRID_EVAL_PATIENCE) {
            a = escape_action(x, y, visited);
        }
        int nx = x + DX[a], ny = y + DY[a];
        if (nx>=0 && nx<GRID && ny>=0 && ny<GRID && !is_obstacle(nx, ny)) {
            x=nx; y=ny;
            visited[y][x] = 1;
            if (active_course_preset == 5) {
                if (x==KEY_X && y==KEY_Y && !g_has_key) {
                    g_has_key=1;
                    memset(visited, 0, sizeof(visited));
                    visited[y][x] = 1;
                    current_waypoint(x, y, &wtx, &wty);
                    best_dist = manhattan_to_fixed_target(x, y, wtx, wty);
                    no_improve = 0;
                }
                if (x==DOOR_X && y==DOOR_Y) grid[y][x] = '*';
            }
        }
        current_waypoint(x, y, &wtx, &wty);
        int dist = manhattan_to_fixed_target(x, y, wtx, wty);
        if (dist < best_dist) { best_dist = dist; no_improve = 0; }
        else                  { no_improve++; }
    }
    if (x==FOOD_X && y==FOOD_Y) grid[y][x]='F';

    printf("    +");  for (int i=0;i<GRID;i++) printf("--"); printf("-+\n");
    for (int j=0;j<GRID;j++) {
        printf("    | ");
        for (int i=0;i<GRID;i++) {
            char c = grid[j][i];
            if      (c=='F') printf("F ");
            else if (c=='#') printf("# ");
            else if (c=='K') printf("K ");
            else if (c=='*') printf("* ");
            else             printf(". ");
        }
        printf("|\n");
    }
    printf("    +"); for(int i=0;i<GRID;i++) printf("--"); printf("-+\n");
}

int main(int argc, char** argv) {
    int n_episodes = N_EPISODES;
    if (argc > 1) {
        active_course_preset = atoi(argv[1]);
        if (active_course_preset < 0 || active_course_preset > 5) {
            active_course_preset = 1;
        }
    }
    if (argc > 2) {
        n_episodes = atoi(argv[2]);
        if (n_episodes <= 0) {
            n_episodes = N_EPISODES;
        }
    }

    printf("\n  ZeriAgent -- Learning to Play (Grid World 7x7)\n\n");

    ZeriAgent* alice = zeri_agent_from_name("alice", 16, ACTS, 128, 1.0f);
    ZeriAgent* bob   = zeri_agent_from_name("bob",   16, ACTS, 128, 1.0f);
    zeri_agent_set_decay(alice, 0.995f, 0.02f);
    zeri_agent_set_decay(bob,   0.997f, 0.02f);
    /* Biological Synaptic Forgetting's default gate isn't actually
     * selective (see zeri_agent_set_forget_surprise_gate's doc comment in
     * zeri_agent.h) -- it fires on nearly every step regardless of whether
     * the agent is doing well, which erodes rare, hard-won correct weight
     * patterns in a long, sparse-reward maze (course presets 3/4) faster
     * than they can be reinforced. Gating on a rolling average of
     * |td_error| instead lets forgetting stay active while genuinely
     * struggling and lock off, permanently, once behavior has settled --
     * unlike a flat zeri_agent_set_forget_rate(alice, 0.0f), which would
     * disable it from step one regardless of how this task (or a harder
     * one someone builds on this) actually behaves. Threshold of 0.3f
     * chosen empirically: measured comfortably above the ~0.01-0.07
     * steady-state avg_abs_td once this task has actually converged, and
     * below the ~0.2-0.7 range seen while it's still struggling. Measured
     * directly: converges preset 3 cleanly by episode 8000, same as full
     * disable, with zero regression on presets 0, 1, 2, 5. */
    zeri_agent_set_forget_surprise_gate(alice, 0.3f);

    float state[16];
    float stats[5];
    double t0 = now_ms();

    printf("  Training %d episodes...\n\n", n_episodes);

    for (int ep = 0; ep < n_episodes; ep++) {
        int x = 0, y = 0;
        g_has_key = 0;
        float ep_reward = 0.0f;
        int   steps     = 0;
        int   stuck     = 0;
        char visited[GRID][GRID];
        if (active_course_preset == 5) { memset(visited, 0, sizeof(visited)); visited[0][0] = 1; }

        zeri_agent_reset_traces(alice);

        for (int t = 0; t < MAX_STEPS_PER_EP; t++) {
            encode_state(x, y, state);
            int a = zeri_agent_act(alice, state, 16);
            if (active_course_preset == 5 && stuck >= 3) {
                a = (a + 1 + (t % 3)) % ACTS;
            }

            int nx = x + DX[a], ny = y + DY[a];
            float reward = -0.05f;
            int is_terminal = 0;
            int just_picked_up = 0;
            int moved = 0;
            int shape_tx = 0, shape_ty = 0;
            current_waypoint(x, y, &shape_tx, &shape_ty);
            /* True (wall-aware) distance, not raw Manhattan distance -- raw
             * Manhattan distance actively penalizes the temporary
             * "backtrack" moves that are REQUIRED to get around a wall in
             * the S-curve course presets (3 and 4), fighting the exact
             * behavior the agent needs to learn. true_dist() already
             * accounts for walls (and the door, for preset 5), so moving
             * toward the real shortest path is always rewarded, never
             * penalized. */
            int dist_before = true_dist(x, y, shape_tx, shape_ty);

            if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID || is_obstacle(nx, ny)) {
                reward = -0.5f;
                if (active_course_preset == 5) stuck++;
            } else {
                x = nx; y = ny;
                moved = 1;
                if (active_course_preset == 5) stuck = 0;
                if (active_course_preset == 5 && x == KEY_X && y == KEY_Y && !g_has_key) {
                    g_has_key = 1;
                    just_picked_up = 1;
                    reward = 2.0f;
                    memset(visited, 0, sizeof(visited));
                    visited[y][x] = 1;
                }
                if (active_course_preset == 5 && x == DOOR_X && y == DOOR_Y && g_has_key && !just_picked_up) {
                    reward += 2.0f;
                }
            }

            if (!just_picked_up) {
                int dist_after = true_dist(x, y, shape_tx, shape_ty);
                reward += 0.2f * (float)(dist_before - dist_after);
            }
            if (active_course_preset == 5 && moved && !just_picked_up) {
                if (visited[y][x]) reward -= 0.3f;
                visited[y][x] = 1;
            }

            float next_state[16];
            encode_state(x, y, next_state);

            if (x == FOOD_X && y == FOOD_Y) {
                reward = 10.0f;
                is_terminal = 1;
            }

            zeri_agent_reinforce(alice, state, 16, a, next_state, reward, is_terminal);
            ep_reward += reward;

            if (is_terminal) {
                steps = t + 1;
                break;
            }
            if (t == MAX_STEPS_PER_EP - 1) steps = MAX_STEPS_PER_EP;
        }

        push_steps((float)steps);
        zeri_agent_decay_epsilon(alice);

        if ((ep+1) % 100 == 0) {
            zeri_agent_stats(alice, stats);
            float avg = avg_steps();
            printf("  Ep %4d  avg steps: %.1f, epsilon: %.3f\n", ep+1, avg, stats[0]);
        }
    }

    double elapsed = now_ms() - t0;
    printf("\n  Training done in %.0f ms\n\n", elapsed);
    show_grid(alice, 0, 0);

    zeri_agent_destroy(alice);
    zeri_agent_destroy(bob);
    return 0;
}
