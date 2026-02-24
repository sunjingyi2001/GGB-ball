import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from joblib import dump
from scipy.spatial.distance import cdist
import gc

class ComputeMatrix(object):
    def __init__(self, name_ds, number, decision, n_categorical, para_k, para_lambda):
        self.name_ds = name_ds
        self.number = number
        self.decision = decision
        self.n_categorical = n_categorical
        self.para_k = para_k
        self.para_lambda = para_lambda
        self.real_att_number = decision - n_categorical

    def GranularBallAnalysis(self):
        print("Running with ball metrics only...")
        # Load data
        data_excel = pd.read_excel(f'../Dataset/{self.name_ds}(0.25).xlsx', header=None)
        df = pd.DataFrame(data_excel)
        print("Raw data shape:", df.shape)

        df_np = df.to_numpy()

        # Identify labeled samples (O_l)
        n = df_np.shape[0]
        O_l = set()
        for ui in range(self.number):
            decision_value = df_np[ui, self.decision]
            if decision_value != '*':
                O_l.add(ui)
        print("O_l:", O_l)

        A = set(range(self.decision))

        # Build class sets (D_i) for labeled samples
        O_d_k = {}
        for ui in O_l:
            decision_value = df_np[ui][self.decision]
            O_d_k.setdefault(decision_value, set()).add(ui)
        D_i = list(O_d_k.values())
        print("D_i:", D_i)

        max_len = max(len(d_i) for d_i in D_i)
        for s in D_i:
            if len(s) == max_len:
                p_0 = len(s) / len(O_l)
                print("p_0:", p_0)

        # Precompute distance matrices (no missing values in attributes)
        distance_cache = {}
        for at in range(self.decision):
            if at < self.n_categorical:
                column_data = df_np[:, at].astype(str)
                dist = (column_data[:, None] != column_data[None, :]).astype(np.float16)
                distance_cache[at] = dist
                del column_data, dist
            else:
                col = df_np[:, at].astype(np.float32)
                rng = np.ptp(col)
                if rng == 0:
                    dist = np.zeros((n, n), dtype=np.float32)
                else:
                    dist = cdist(col[:, None], col[:, None], metric='euclidean') / (rng + 1e-10)  # 添加 epsilon
                distance_cache[at] = dist
                del col, dist
            gc.collect()

        def rho_B(P: set):
            block_size = 8
            sq_dist_matrix = np.zeros((n, n), dtype=np.float32)
            P_list = list(P)

            for i in range(0, len(P), block_size):
                P_block = P_list[i:i + block_size]
                mats = [distance_cache[at] for at in P_block]

                # S_P 对应 Student-t kernel: 1 / (1 + φ^2)
                S_P = 1 / (1 + np.stack(mats, axis=0) ** 2)

                # D_attr 对应 (1 - S_a)^2
                D_attr = (1 - S_P) ** 2

                sq_dist_matrix += D_attr.sum(axis=0)
                del mats, S_P, D_attr
                gc.collect()
            dist_matrix = np.sqrt(sq_dist_matrix).astype(np.float16)
            return dist_matrix

        dist = rho_B(A)
        print("Distance matrix sample:", dist[:5, :5])

        def separate_ball(para_p, max_depth=100):
            queue = [(list(O_l), 0)]
            final_balls = []
            while queue:
                idx, depth = queue.pop(0)
                print(f"Processing ball with {len(idx)} samples at depth {depth}")
                if len(idx) <= 1 or depth >= max_depth:
                    final_balls.append((idx, None, 1.0))
                    continue
                submat = dist[np.ix_(idx, idx)]
                i_sub, j_sub = np.unravel_index(submat.argmax(), submat.shape)
                c_idx = idx[i_sub]
                c_prime_idx = idx[j_sub]
                GGB1 = [t for t in idx if dist[t, c_idx] < dist[t, c_prime_idx]]
                GGB2 = [t for t in idx if dist[t, c_idx] >= dist[t, c_prime_idx]]
                del submat
                gc.collect()

                for ggb in (GGB1, GGB2):
                    if not ggb:
                        continue
                    inter_sizes = [len(set(ggb) & Di) for Di in D_i]
                    best_i = int(np.argmax(inter_sizes))
                    purity = inter_sizes[best_i] / len(ggb)
                    if purity >= para_p or len(ggb) <= 2:
                        final_balls.append((ggb))
                    else:
                        queue.append((ggb, depth + 1))
            return final_balls

        # Reduce p values and experiment repetitions
        epsilon = 1e-9
        adjusted_start = p_0 + epsilon
        adjusted_end = 0.9999
        p_list = np.linspace(adjusted_start, adjusted_end, 10).tolist()

        results = {
            'p': [],
            'num_balls': [],
            'avg_ball_size': [],
            'num_balls_std': [],
            'avg_ball_size_std': []
        }

        ball_cache = {}
        for p in p_list:
            num_balls_list = []
            avg_size_list = []
            for _ in range(3):
                if p not in ball_cache:
                    balls = separate_ball(p)
                    ball_cache[p] = balls
                else:
                    balls = ball_cache[p]
                w = len(balls)
                avg_size = len(O_l) / w if w > 0 else 0
                num_balls_list.append(w)
                avg_size_list.append(avg_size)
            results['p'].append(p)
            results['num_balls'].append(np.mean(num_balls_list))
            results['avg_ball_size'].append(np.mean(avg_size_list))
            results['num_balls_std'].append(np.std(num_balls_list))
            results['avg_ball_size_std'].append(np.std(avg_size_list))

        df = pd.DataFrame(results)
        print("Results:\n", df)

        plt.figure(figsize=(10, 6))
        main_color = '#00008B'
        second_color = '#006400'
        sns.lineplot(data=df, x='p', y='num_balls',
                     marker='o', label=r'$\omega$', color=main_color,linewidth=2.5, markersize=8)
        plt.fill_between(df['p'],
                         df['num_balls'] - df['num_balls_std'],
                         df['num_balls'] + df['num_balls_std'],
                         color=main_color, alpha=0.2)
        sns.lineplot(data=df, x='p', y='avg_ball_size',
                     marker='o', label=r'$\bar{\omega}=\frac{\sum_{j=1}^{\omega}|GGB_{p}^{(j)}|}{\omega}$',
                     color=second_color,linewidth=2.5, markersize=8)
        plt.fill_between(df['p'],
                         df['avg_ball_size'] - df['avg_ball_size_std'],
                         df['avg_ball_size'] + df['avg_ball_size_std'],
                         color=second_color, alpha=0.2)

        target_size = 2
        # Restrict optimal_p_size to values less than 1
        df_filtered = df[df['p'] < 1]
        if not df_filtered.empty:
            optimal_p_size = df_filtered.iloc[(df_filtered['avg_ball_size'] - target_size).abs().argmin()]['p']
        else:
            optimal_p_size = df['p'].iloc[0]  # Fallback to first p if all are filtered
        optimal_num_balls = df.iloc[(df['p'] - optimal_p_size).abs().argmin()]['num_balls']
        optimal_avg_size = df.iloc[(df['p'] - optimal_p_size).abs().argmin()]['avg_ball_size']
        plt.scatter(optimal_p_size, optimal_num_balls, color='green', s=30, zorder=5)
        plt.scatter(optimal_p_size, optimal_avg_size, color='green', s=30, zorder=5)


        plt.annotate(f'p={optimal_p_size:.3f}',
                     xy=(optimal_p_size, optimal_num_balls),
                     xytext=(5, 5),  # Offset in points
                     textcoords='offset points',
                     fontsize=18, color='green', zorder=6)


        plt.annotate(f'p={optimal_p_size:.3f}',
                     xy=(optimal_p_size, optimal_avg_size),
                     xytext=(5, 5),  # Offset in points
                     textcoords='offset points',
                     fontsize=18, color='green', zorder=6)

        plt.xlabel('p', fontsize=18, fontweight='bold')
        plt.ylabel('Count / Size', fontsize=18, fontweight='bold')
        # 3. 设置坐标轴刻度字体大小
        plt.xticks(p_list, fontsize=18, rotation=45)  # 如果 p 很多，旋转一下更好看
        plt.yticks(fontsize=16)

        # 4. 设置图例字体大小
        # prop 参数专门控制图例中的字体
        plt.legend(fontsize=16, loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(f'../GrcBALLtu/{self.name_ds}_ball_metrics_vs_purity.png')
        # plt.show()
        optimal_balls = ball_cache[optimal_p_size]
        balls_save_dir = '../GranularBalls'
        os.makedirs(balls_save_dir, exist_ok=True)
        dump(optimal_balls, os.path.join(balls_save_dir, f'{self.name_ds}_optimal_balls.joblib'))

        print(f"最优 p 的分球结果已保存到 {os.path.join(balls_save_dir, f'{self.name_ds}_optimal_balls.joblib')}")

        EndTime = datetime.datetime.today()
        print(f"从 {EndTime} 结束粒球分析")
        print(f"分析总共花费时长为 {EndTime - StartTime}")


if __name__ == '__main__':
    StartTime = datetime.datetime.today()
    print(f"从 {StartTime} 开始GranularBallAnalysis")
    dic_ds = {

        # '000_example': ['000_example', 10, 6,4, 100, 100],

    }
    for v in dic_ds.values():
        ds = ComputeMatrix(*v)
        ds.GranularBallAnalysis()