# Hybrid IDO+SB3 Agent Benchmark Results

**Configuration**: 3 episodes × 100 steps per task

## Summary Table

| Task | Agent | Avg Return | Avg Speed | Avg η | Avg Steps | NV Total |
|------|-------|-----------|-----------|-------|-----------|----------|
| cheetah-run | IDO | 0.8295 | 0.1162 | 9.848932 | 100.0 | 0 |
| cheetah-run | PPO | 19.0054 | 1.9005 | 0.000000 | 100.0 | 0 |
| cheetah-run | Hybrid-PPO | 17.8626 | 1.7863 | 8.155706 | 100.0 | 0 |
| walker-walk | IDO | 6.0646 | 0.1679 | 1.445518 | 100.0 | 0 |
| walker-walk | PPO | 16.5218 | 0.5641 | 0.000000 | 100.0 | 0 |
| walker-walk | SAC | 46.2171 | 0.8105 | 0.000000 | 100.0 | 0 |
| walker-walk | Hybrid-PPO | 18.9298 | 0.5760 | 1.042618 | 100.0 | 0 |
| walker-walk | Hybrid-SAC | 65.6262 | 0.8867 | 0.533664 | 100.0 | 0 |
| humanoid-stand | IDO | 8.0549 | 0.0971 | 2.886428 | 100.0 | 118 |
| humanoid-stand | PPO | 5.6673 | 0.2591 | 0.000000 | 100.0 | 0 |
| humanoid-stand | SAC | 35.8201 | 0.4345 | 0.000000 | 100.0 | 0 |
| humanoid-stand | Hybrid-PPO | 5.8845 | 0.2070 | 13.379565 | 100.0 | 144 |
| humanoid-stand | Hybrid-SAC | 29.2194 | 0.3109 | 7.621665 | 100.0 | 48 |

## Detailed Results

### cheetah-run

| Metric | IDO | PPO | SAC | Hybrid-PPO | Hybrid-SAC |
|--------|-----|-----|-----|------------|------------|
| Avg Return | 0.8295 | 19.0054 | — | 17.8626 | — |
| Std Return | 0.1269 | 0.3829 | — | 2.7169 | — |
| Avg Speed | 0.1162 | 1.9005 | — | 1.7863 | — |
| Std Speed | 0.0156 | 0.0383 | — | 0.2717 | — |
| Avg η | 9.848932 | 0.000000 | — | 8.155706 | — |
| Std η | 0.010373 | 0.000000 | — | 0.261273 | — |
| Avg Steps | 100.0 | 100.0 | — | 100.0 | — |
| NV Total | 0 | 0 | — | 0 | — |
| Avg Time (s) | 0.50 | 0.12 | — | 0.84 | — |

**Hybrid-PPO Mode Distribution** (cheetah-run):
- EXPLOIT: 300 steps (100.0%)
- EXPLORE: 0 steps (0.0%)
- SAFE: 0 steps (0.0%)

### walker-walk

| Metric | IDO | PPO | SAC | Hybrid-PPO | Hybrid-SAC |
|--------|-----|-----|-----|------------|------------|
| Avg Return | 6.0646 | 16.5218 | 46.2171 | 18.9298 | 65.6262 |
| Std Return | 2.2087 | 4.1442 | 8.6336 | 6.0806 | 6.2482 |
| Avg Speed | 0.1679 | 0.5641 | 0.8105 | 0.5760 | 0.8867 |
| Std Speed | 0.1257 | 0.0490 | 0.0981 | 0.1186 | 0.0582 |
| Avg η | 1.445518 | 0.000000 | 0.000000 | 1.042618 | 0.533664 |
| Std η | 0.216473 | 0.000000 | 0.000000 | 0.217287 | 0.212135 |
| Avg Steps | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| NV Total | 0 | 0 | 0 | 0 | 0 |
| Avg Time (s) | 0.50 | 0.22 | 0.17 | 0.89 | 1.00 |

**Hybrid-PPO Mode Distribution** (walker-walk):
- EXPLOIT: 282 steps (94.0%)
- EXPLORE: 0 steps (0.0%)
- SAFE: 18 steps (6.0%)

**Hybrid-SAC Mode Distribution** (walker-walk):
- EXPLOIT: 238 steps (79.3%)
- EXPLORE: 0 steps (0.0%)
- SAFE: 62 steps (20.7%)

### humanoid-stand

| Metric | IDO | PPO | SAC | Hybrid-PPO | Hybrid-SAC |
|--------|-----|-----|-----|------------|------------|
| Avg Return | 8.0549 | 5.6673 | 35.8201 | 5.8845 | 29.2194 |
| Std Return | 4.9174 | 2.7277 | 4.6618 | 0.9411 | 20.6705 |
| Avg Speed | 0.0971 | 0.2591 | 0.4345 | 0.2070 | 0.3109 |
| Std Speed | 0.0275 | 0.0480 | 0.1461 | 0.0739 | 0.0326 |
| Avg η | 2.886428 | 0.000000 | 0.000000 | 13.379565 | 7.621665 |
| Std η | 0.170586 | 0.000000 | 0.000000 | 6.227496 | 0.354938 |
| Avg Steps | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| NV Total | 118 | 0 | 0 | 144 | 48 |
| Avg Time (s) | 0.43 | 0.29 | 0.35 | 1.05 | 0.98 |

**Hybrid-PPO Mode Distribution** (humanoid-stand):
- EXPLOIT: 49 steps (16.3%)
- EXPLORE: 45 steps (15.0%)
- SAFE: 206 steps (68.7%)

**Hybrid-SAC Mode Distribution** (humanoid-stand):
- EXPLOIT: 82 steps (27.3%)
- EXPLORE: 87 steps (29.0%)
- SAFE: 131 steps (43.7%)

## Bug/Issue Report

Any errors or issues encountered during verification:

### No Issues Found ✅

All agents ran successfully across all tasks.
