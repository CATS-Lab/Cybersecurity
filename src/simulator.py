from lower_level import LowerLevelProblem
from parameters import *
import time
import os


class CFSimulator:
    def __init__(self, attack_time, attack_a, ref_trajectory, id=1):
        # Initialize simulation variables
        a_l, v_l, p_l, a_f_init, v_f_init, p_f_init = ref_trajectory
        self.ah_l = np.copy(a_l)
        self.ah_l[attack_time:] = attack_a[attack_time - Nd - Np:]  # attacked acceleration

        self.a = np.zeros((2, T))
        self.a[0] = np.copy(a_l)                     # leader acceleration
        self.a[1, 0:Nd + 1] = a_f_init               # follower initial acceleration

        self.v = np.zeros((2, T))
        self.v[0] = np.copy(v_l)                     # leader velocity
        self.v[1, 0:Nd + 1] = v_f_init               # follower initial velocity

        self.p = np.zeros((2, T))
        self.p[0] = np.copy(p_l)                     # leader position
        self.p[1, 0:Nd + 1] = p_f_init               # follower initial position

        # Generate the attacked trajectory (ph_l, vh_l)
        self.vh_l = np.zeros(T)
        self.vh_l[0] = v_l[0]
        self.ph_l = np.zeros(T)
        self.ph_l[0] = p_l[0]
        for step in range(1, T):
            self.vh_l[step] = self.vh_l[step - 1] + self.ah_l[step - 1] * dt
            self.ph_l[step] = self.ph_l[step - 1] + (self.vh_l[step - 1] + self.vh_l[step]) / 2 * dt

        self.TTCs = None
        self.modes = None
        self.safety_headway = None
        self.times = None
        self.id = id

    def simulate(self):
        self.TTCs = np.full(T, M, dtype='float64')
        self.modes = [""] * (Nd + 1)
        self.safety_headway = np.full(T, M, dtype='float64')
        last_mode = 'connected'
        last_am = 0
        last_protected_time = 0
        self.times = []

        # Run simulation
        for step in range(Nd + 1, T - Np - 1):
            self.update_v(step)
            self.update_p(step)

            lp = LowerLevelProblem(
                self.ah_l[step - Nd - 1],
                self.ah_l[step - Nc + Ns - 1:step - Nc + Ns + 1],
                self.a[0, step - Nd - 1],
                self.vh_l[step - Nc + 1:step - Nc + Np + 2],
                self.ph_l[step - Nc + 1:step - Nc + Np + 2],
                self.v[0, step - Nd],
                self.p[0, step - Nd],
                self.a[1, step - 1],
                self.v[1, step],
                self.p[1, step],
                last_mode,
                last_am,
                last_protected_time
            )

            # Use combined control strategy
            self.a[1, step], last_mode, last_am, last_protected_time, headway = lp.combined_strategy()

            # Alternative strategies:
            # self.a[1, step], last_mode, last_am, last_protected_time, headway = lp.no_security_strategy_connected()
            # self.a[1, step], last_mode, last_am, last_protected_time, headway = lp.compare()

            self.TTCs[step] = self.calculate_TTC(self.v[:, step], self.p[:, step])

            if self.TTCs[step] == 0:
                self.modes.append('Crash')
                continue

            self.modes.append(last_mode)
            self.safety_headway[step + 1] = headway

        self.modes = self.modes + [""] * (Np + 1)

    def update_v(self, step):
        self.v[:, step] = self.v[:, step - 1] + self.a[:, step - 1] * dt

    def update_p(self, step):
        self.p[:, step] = self.p[:, step - 1] + (self.v[:, step - 1] + self.v[:, step]) / 2 * dt

    def calculate_TTC(self, v, x):
        delta_x = x[0] - x[1]
        delta_v = v[1] - v[0]
        if delta_x <= 0:
            ttc = 0
        elif delta_v <= 0:
            ttc = M
        else:
            ttc = delta_x / delta_v
            ttc = min(ttc, M)
        return ttc

    def print_cars(self, step):
        msg = f'step: {step}; '
        msg += f'LV_x: {self.p[0, step]}, FV_x: {self.p[1, step]}, '
        msg += f'LV_v: {self.v[0, step]}, FV_v: {self.v[1, step]}; '
        msg += f'ttc: {self.TTCs[step]};'
        print(msg)

    def print_trajectory(self, file, print_time=False, print_sensitivity=False):
        time_index = np.arange(len(self.a[0])) * dt

        df = pd.DataFrame({
            'Time_Index': time_index,
            'Pos_LV': self.p[0],
            'Speed_LV': self.v[0],
            'Acc_LV': self.a[0],
            'Pos_FV': self.p[1],
            'Speed_FV': self.v[1],
            'Acc_FV': self.a[1],
            'Pos_attack': self.ph_l,
            'Speed_attack': self.vh_l,
            'Acc_attack': self.ah_l,
            'TTC': self.TTCs,
            'Mode': self.modes,
            'Safety': self.safety_headway,
        })

        df['headway'] = df['Pos_LV'] - df['Pos_FV']
        df['time_headway'] = df['headway'] / df['Speed_FV']

        df.to_csv(file, index=False)

        if print_time:
            file_path = './results/runtime.csv'
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                df_prev = pd.read_csv(file_path)
            else:
                df_prev = pd.DataFrame()
            df_prev[f'Time{self.id}'] = self.times
            df_prev.to_csv(file_path, index=False)

        if print_sensitivity:
            connected_df = df[df['Mode'] == 'connected']
            safety_mean_connected = connected_df['Safety'].mean()
            headway_mean_connected = connected_df['headway'].mean()
            time_headway_mean_connected = connected_df['time_headway'].mean()

            filtered_df = df[((df['Time_Index'] >= Nd * dt + 1) &
                              (df['Time_Index'] <= (T - Np) * dt - 2))]
            headway_mean_filtered = filtered_df['headway'].mean()
            time_headway_mean_filtered = filtered_df['time_headway'].mean()

            min_ttc = df['TTC'].min()

            new_row = {
                # 'Safety_Mean_Connected': safety_mean_connected,
                # 'Headway_Mean_Connected': headway_mean_connected,
                # 'Time_Headway_Mean_Connected': time_headway_mean_connected,
                'Min_TTC': min_ttc,
                'Headway_Mean_Filtered': headway_mean_filtered,
                'Time_Headway_Mean_Filtered': time_headway_mean_filtered,
            }

            output_file_path = './results/sensitivity.csv'
            try:
                output_df = pd.read_csv(output_file_path)
            except FileNotFoundError:
                output_df = pd.DataFrame(columns=new_row.keys())

            new_row_df = pd.DataFrame([new_row])
            output_df = pd.concat([output_df, new_row_df], ignore_index=True)
            output_df.to_csv(output_file_path, index=False)
