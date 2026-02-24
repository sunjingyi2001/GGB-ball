import pandas as pd
import numpy as np
import datetime
import os
import openpyxl
import pickle
import math
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
import multiprocessing
from sklearn.neighbors import KDTree
from functools import lru_cache

class ComputeMatrix(object):
    # 数据集名：[数据集名，数据集样本数，条件属性总数（分类加实值），分类属性总数，k，lambda]
    def __init__(self, name_ds, number, decision, n_categorical):
        self.name_ds = name_ds
        self.number = number
        self.decision = decision
        self.n_categorical = n_categorical
        self.real_att_number = decision - n_categorical

    def AttributeReductionBALL(self):


            # 加载最优 p 的分球结果
        balls_save_dir = '../GranularBalls'
        balls_path = os.path.join(balls_save_dir, f'{self.name_ds}_optimal_balls.joblib')
        if os.path.exists(balls_path):
            with open(balls_path, 'rb') as f:
                GGB_ALL = pickle.load(f)
                print("GGB_ALL",GGB_ALL)
            print(f"加载到最优 p 的分球结果: {GGB_ALL}")
        else:
            print(f"警告：未找到 {balls_path}，无法加载分球结果，请先运行 GranularBallAnalysis")
            return  # 或者根据需要重新计算



        # Load data
        data_excel = pd.read_excel(f'../Dataset/{self.name_ds}(0.25).xlsx', header=None)
        df = pd.DataFrame(data_excel)

        # 调试：打印原始数据类型
        print("Raw data types:", df.dtypes)

        # 处理分类属性（0 到 n_categorical-1）
        for col in range(self.n_categorical):
            df[col] = df[col].astype(str).replace(['nan', 'NaN', ''], 'missing')
            df[col] = df[col].astype('category')

        # 处理数值属性（n_categorical 到 decision-1）
        for col in range(self.n_categorical, self.decision):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                print(f"Warning: NaN found in numerical column {col}, filling with mean")
                df[col].fillna(df[col].mean(), inplace=True)
            df[col] = df[col].astype(np.float32)

        # 处理决策列
        df[self.decision] = df[self.decision].astype(str).replace('*', np.nan).astype('category')

        df_np = df.to_numpy()
        print(f"Processed df:\n{df_np}")
        print("Processed data types:", df.dtypes)

        n = df_np.shape[0]
        O_l = set()
        for ui in range(self.number):
            decision_value = df_np[ui, self.decision]
            if not pd.isna(decision_value):
                O_l.add(ui)
        print("O_l", O_l)
        O_l_ = list(O_l)

        O_u = set()
        for ui in range(self.number):
            decision_value = df_np[ui, self.decision]
            if pd.isna(decision_value):
                O_u.add(ui)
        print("O_u", O_u)
        O_u_ = list(O_u)

        theta = len(O_u) / self.number
        A = set(range(self.decision))

        O_d_k = {}
        for ui in range(self.number):
            decision_value = df_np[ui][self.decision]
            if pd.isna(decision_value):
                continue
            O_d_k.setdefault(decision_value, set()).add(ui)
        D_i = list(O_d_k.values())


        distance_cache = {}

        def distance_matrix_categorical(at):
            if at not in distance_cache:
                column_data = df_np[:, at]
                column_data = column_data.astype(str)
                dist = (column_data[:, None] != column_data[None, :]).astype(np.float32)
                # print(f"Shape of distance matrix for categorical attribute {at}: {dist.shape}")
                distance_cache[at] = dist
            return distance_cache[at]

        def distance_matrix_real(at):
            if at not in distance_cache:
                col = df_np[:, at].astype(np.float32)
                rng = np.ptp(col)
                if rng == 0:
                    dist = np.zeros((len(col), len(col)), dtype=np.float32)
                else:
                    abs_diff = np.abs(col[:, None] - col[None, :])
                    dist = abs_diff / rng
                # print(f"Shape of distance matrix for real attribute {at}: {dist.shape}")
                distance_cache[at] = dist
            return distance_cache[at]

        def rho_a(P: set):
            mats = Parallel(n_jobs=1)(
                delayed(distance_matrix_categorical)(at) if at < self.n_categorical
                else delayed(distance_matrix_real)(at)
                for at in P
            )
            expected_shape = (n, n)
            shapes = [mat.shape for mat in mats]
            if any(shape != expected_shape for shape in shapes):
                raise ValueError(f"Inconsistent matrix shapes in rho_a for attributes {list(P)}: {shapes}")
            S_P = 1 / (1 + np.stack(mats, axis=0) ** 2)
            D_attr = (1 - S_P)
            return D_attr

        def rho_B(P):
            D_attr2 = rho_a(P) ** 2
            dist_matrix = np.sqrt(D_attr2.sum(axis=0))
            return dist_matrix



        def M_R_alpha(P,  alpha):
            # print(">>> M_R_alpha 收到的 P =", P)
            rho_matrices = rho_a(P)
            # print("rho_matrices", rho_matrices)
            prop_mask = np.logical_and.reduce([D <= alpha for D in rho_matrices])
            # print("prop_mask", prop_mask)
            n = prop_mask.shape[0]
            relation = np.zeros((n, n), dtype=int)
            for grp in GGB_ALL:
                mask_grp = np.zeros((n, n), dtype=bool)
                mask_grp[np.ix_(grp, grp)] = True
                valid = mask_grp & prop_mask
                relation[valid] = 1
            return relation


        def GGB_B_p_for_X(P,  alpha, lam, S):

            M_P = M_R_alpha(P,  alpha)
            # print("M_P", M_P)
            result = set()
            for ball in GGB_ALL:
                G = set(ball)
                sizeG = len(G)
                if sizeG == 0:
                    continue
                for t in G:
                    B_t_G = {j for j in G if M_P[t, j] == 1}
                    # print("B_t_G", B_t_G)
                    if not (B_t_G <= S):
                        continue
                    if len(B_t_G & S) / sizeG >= lam:
                        result.add(t)
            return result

        def Dep(P, alpha, lam):
            pos = set()
            for S in D_i:
                pos |= GGB_B_p_for_X(P, alpha, lam, S)
            Dep = len(pos) / len(O_l)
            return Dep

        def S_B(P, alpha):
            S_B = rho_B(P)
            dist_u = S_B[np.ix_(O_u_, O_u_)]
            # print("dist_u", dist_u)
            S_u = (dist_u <= alpha).astype(int)
            return S_u
        print("S_B({0,1,2},0.6)", S_B({0, 1, 2}, 0.6))

        def GE_B(P, alpha):
            S_u = S_B(P, alpha)
            # print("S_u", S_u)
            row_sums = S_u.sum(axis=1)
            # print("row_sums", row_sums)
            p = row_sums / len(O_u)
            # print("p", p)
            GE_u = - np.mean(np.log(p + 1e-10))  # 避免 log(0)
            # print("GE^u_alpha(B) =", GE_u)
            return GE_u

        def GGB_imp(P, alpha, lam):
            num1 = Dep(P, alpha, lam)
            den1 = Dep(A, alpha, lam)
            if den1 == 0:
                ratio1 = 0
            else:
                ratio1 = num1 / den1
            num2 = GE_B(P, alpha)
            den2 = GE_B(A, alpha)
            if den2 == 0:
                ratio2 = 0
            else:
                ratio2 = num2 / den2
            return (1 - theta) * ratio1 + theta * ratio2

        def AttributeReduction( alpha, lam):
            GGB_imp_A = GGB_imp(A, alpha, lam)
            red = set()
            for at in range(self.decision):
                red.add(at)
                if GGB_imp(red, alpha, lam) == GGB_imp_A:
                    return red
            return red

        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        lambda_list = [0.5, 0.6, 0.7, 0.8, 0.9]
        # print('lambda_list', lambda_list)


        Reduct = {}
        for a in alpha:
            Reduct[a] = {}
            for lam in lambda_list:
                Reduct[a][lam] = AttributeReduction(a, lam)

        print("Reduct", Reduct)
        folder_path = '../Reduction02/'
        os.makedirs(folder_path, exist_ok=True)
        with open('../Reduction02/' + self.name_ds + '_TheReduction.pkl', 'wb') as file:
            pickle.dump(Reduct, file)
        EndTime = datetime.datetime.today()
        print("从 {} 结束计算分类精度".format(EndTime))
        print("利用AlgorithmBALL计算约简集总共花费时长为{}".format(EndTime - StartTime))


if __name__ == '__main__':
    # print("%s 开始NumericalExperiment")
    StartTime = datetime.datetime.today()
    print("从 {} 开始NumericalExperiment".format(StartTime))
    dic_ds = {

        # '000_example': ['000_example', 10, 6,4],

    }
    for v in dic_ds.values():
        ds = ComputeMatrix(*v)
        ds.AttributeReductionBALL()