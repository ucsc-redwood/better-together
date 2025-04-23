#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import json
import hashlib
from z3 import Optimize, Bool, Real, Sum, If, Or, Not, RealVal, sat, Implies, And

def load_csv_and_compute_averages(csv_path):
    """
    Load data from a CSV file and compute average timings for each stage across all runs.
    Returns a list of lists: [ [little, medium, big, vulkan], ... ] for each stage.
    """
    df = pd.read_csv(csv_path)
    avg_df = df.groupby("stage")[['little', 'medium', 'big', 'vulkan']].mean()

    avg_timings = []
    # assume stages numbered from min to max present
    stages = sorted(avg_df.index.astype(int).tolist())
    for stage in stages:
        row = avg_df.loc[stage]
        avg_timings.append([row['little'], row['medium'], row['big'], row['vulkan']])
    return avg_timings, stages


def define_data(stage_timings):
    num_stages = len(stage_timings)
    core_types = ['Little', 'Medium', 'Big', 'GPU']
    return num_stages, core_types, stage_timings


def create_decision_vars(num_stages, core_types):
    x = {}
    for i in range(num_stages):
        for c in core_types:
            x[(i, c)] = Bool(f"x_{i}_{c}")
    return x


def add_constraints(opt, x, num_stages, core_types, stage_timings):
    # assignment: exactly one PU per stage
    for i in range(num_stages):
        opt.add(Or([x[(i, c)] for c in core_types]))
        for j in range(len(core_types)):
            for k in range(j+1, len(core_types)):
                opt.add(Or(Not(x[(i, core_types[j])]), Not(x[(i, core_types[k])])))
    # contiguity: each PU's assigned stages form contiguous block(s)
    for c in core_types:
        for i in range(num_stages):
            for j in range(i+1, num_stages):
                for k in range(j+1, num_stages):
                    opt.add(Implies(And(x[(i,c)], x[(k,c)]), x[(j,c)]))


def add_timing_opt(opt, x, num_stages, core_types, stage_timings):
    T_max = Real('T_max')
    T_min = Real('T_min')
    Gapness = Real('Gapness')
    opt.add(T_max > 0, T_min > 0)

    # for each possible contiguous segment and core, bound T_max, T_min
    for c in core_types:
        idx = core_types.index(c)
        for i in range(num_stages):
            for j in range(i, num_stages):
                seg = And([x[(k,c)] for k in range(i, j+1)])
                seg_sum = Sum([RealVal(stage_timings[k][idx]) for k in range(i, j+1)])
                opt.add(Implies(seg, seg_sum <= T_max))
                is_start = i==0 or Not(x[(i-1,c)])
                is_end = j==num_stages-1 or Not(x[(j+1,c)])
                max_seg = And(seg, is_start, is_end)
                opt.add(Implies(max_seg, seg_sum >= T_min))
    opt.add(Gapness == T_max - T_min)
    opt.minimize(Gapness)
    return T_max, T_min, Gapness


def block_solution(opt, x, num_stages, core_types, model):
    block = []
    for i in range(num_stages):
        for c in core_types:
            if model.evaluate(x[(i,c)]): block.append(Not(x[(i,c)]))
            else: block.append(x[(i,c)])
    opt.add(Or(block))


def extract_solution(model, x, num_stages, core_types, stage_timings):
    # build assignments and chunks
    assigns = {i: None for i in range(num_stages)}
    for i in range(num_stages):
        for c in core_types:
            if model.evaluate(x[(i,c)]): assigns[i] = c
    # chunk segmentation
    chunks = []
    curr = assigns[0]
    stages = [0]
    for i in range(1, num_stages):
        if assigns[i] == curr: stages.append(i)
        else:
            chunks.append({'core_type':curr, 'stages':stages.copy(),
                           'time': sum(stage_timings[s][core_types.index(curr)] for s in stages)})
            curr = assigns[i]
            stages = [i]
    chunks.append({'core_type':curr, 'stages':stages.copy(),
                   'time': sum(stage_timings[s][core_types.index(curr)] for s in stages)})
    times = [ch['time'] for ch in chunks]
    max_t, min_t = max(times), min(times)
    avg_t = sum(times)/len(times)
    ratio = min_t/max_t if max_t>0 else 0
    # UID
    summary = ''.join(ch['core_type'][0] + str(len(ch['stages'])) for ch in chunks)
    gap_str = f"{(max_t-min_t):.2f}".replace('.', '')
    uid = f"SCH-{summary}-G{gap_str}-{hashlib.md5(str(chunks).encode()).hexdigest()[:4]}"
    return {
        'uid': uid,
        'chunks': chunks,
        'metrics': {'max_time': max_t, 'min_time': min_t,
                    'gapness': max_t-min_t,
                    'ratio': ratio, 'avg_time': avg_t},
        'assignments': assigns
    }


def solve(stage_timings, num_solutions):
    num_stages, core_types, stage_timings = define_data(stage_timings)
    opt = Optimize()
    x = create_decision_vars(num_stages, core_types)
    add_constraints(opt, x, num_stages, core_types, stage_timings)
    T_max, T_min, Gapness = add_timing_opt(opt, x, num_stages, core_types, stage_timings)

    solutions = []
    count = 0
    while count < num_solutions and opt.check() == sat:
        m = opt.model()
        sol = extract_solution(m, x, num_stages, core_types, stage_timings)
        solutions.append(sol)
        block_solution(opt, x, num_stages, core_types, m)
        count += 1
    return solutions


def main():
    parser = argparse.ArgumentParser(description='Schedule solver from CSV folder')
    parser.add_argument('--csv_folder', required=True, help='Folder with device_app_backend_{normal,fully}.csv')
    parser.add_argument('--device', required=True)
    parser.add_argument('--app', required=True)
    parser.add_argument('--backend', required=True, choices=['vk','cu'])
    parser.add_argument('--num_solutions', type=int, required=True)
    parser.add_argument('--output_folder', required=True, help='Root output for JSON schedules')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    backend = args.backend
    device = args.device
    app = args.app
    nsol = args.num_solutions
    csv_folder = args.csv_folder

    for kind in ['normal', 'fully']:
        csv_path = os.path.join(csv_folder, f"{device}_{app}_{backend}_{kind}.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping missing CSV: {csv_path}")
            continue
        stage_timings, stages = load_csv_and_compute_averages(csv_path)
        sols = solve(stage_timings, nsol)
        out_path = os.path.join(
            args.output_folder,
            f"{device}_{app}_{backend}_{kind}_schedules.json"
        )
        with open(out_path, 'w') as f:
            json.dump(sols, f, indent=2)
        print(f"Wrote {len(sols)} solutions for {kind} to {out_path}")

if __name__ == '__main__':
    main()
