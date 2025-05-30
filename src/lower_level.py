from parameters import *
from gurobipy import Model, GRB, QuadExpr
import sys


class LowerLevelProblem:
    def __init__(self, ah_l_p, ah_l_f, ad_l, vh_l, ph_l, vd_l, pd_l, a_f, v_f, p_f, last_mode, last_am,
                 last_protected_time):
        self.mode = None
        self.ah_l_p = ah_l_p
        self.ah_l_f = ah_l_f
        self.ad_l = ad_l
        self.vh_l = vh_l
        self.ph_l = ph_l
        self.vd_l = vd_l
        self.pd_l = pd_l
        self.a_f = a_f
        self.v_f = v_f
        self.p_f = p_f
        self.last_mode = last_mode
        self.protected_time = last_protected_time
        self.am = last_am

    def combined_strategy(self):
        a = 0
        mode = self.last_mode

        if self.last_mode == "connected":
            if self.ah_l_f[1] > eta1 or self.ah_l_f[1] - self.ah_l_f[0] > eta2:
                mode = "protected"
                self.protected_time = 1
            if abs(self.ad_l - self.ah_l_p) > 1e-2:  # !=
                mode = "emergent"
        elif self.last_mode == "protected":
            if abs(self.ad_l - self.ah_l_p) > 1e-2:  # !=
                mode = "emergent"
            elif self.protected_time > Ns + Nd + 1:
                mode = "connected"
        elif self.last_mode == "emergent":
            headway = self.get_non_connected_safety_headway()
            if self.pd_l - self.p_f >= headway:
                mode = "non-connected"
            else:
                mode = "emergent"

        if mode == "emergent":
            headway = self.get_non_connected_safety_headway()
            if self.pd_l - self.p_f >= headway:
                mode = "non-connected"
            else:
                mode = "emergent"

        h = 0
        if mode == 'connected':
            a, h = self.connected_strategy()
        elif mode == 'non-connected':
            a, h = self.non_connected_strategy()
        elif mode == 'protected':
            a, h = self.protected_strategy()
        elif mode == 'emergent':
            a = a_range[0]
            h = self.get_non_connected_safety_headway()

        return a, mode, self.am, self.protected_time, h

    def non_connected_strategy(self):
        model = Model("Non-connected model")
        model.setParam('OutputFlag', False)

        # Create variables
        a0 = model.addVar(lb=a_range[0], ub=a_range[1], name="a1")
        v1 = model.addVar(lb=v_range[0], ub=v_range[1], name="v1")
        p1 = model.addVar(lb=self.p_f, ub=GRB.INFINITY, name="p1")
        h1 = model.addVar(lb=0, ub=GRB.INFINITY, name="h1")
        q = model.addVars(3, vtype=GRB.BINARY, name="q")

        # Add safety constraints
        model.addConstr(M * (1 - q[0]) >= self.vd_l - v1, name="q1")
        model.addQConstr(h1 + M * (1 - q[0]) >= (Nd * dt) * v1 - v1 * v1 / 2 / a_range[0], name="headway1")
        model.addConstr(M * (1 - q[1]) >= v1 - self.vd_l, name="q2_1")
        model.addConstr(M * (1 - q[1]) >= self.vd_l - v1 - (Nd * dt) * a_range[1], name="q2_2")
        model.addQConstr(
            h1 + M * (1 - q[1]) >= (Nd * dt) * self.vd_l - (self.vd_l - v1) ** 2 / (2 * a_range[1]) - self.vd_l ** 2 / (
                    2 * a_range[0]), name="headway2")
        model.addConstr(M * (1 - q[2]) >= v1 + (Nd * dt) * a_range[1] - self.vd_l, name="q3")
        model.addQConstr(
            h1 + M * (1 - q[2]) >= (Nd * dt) * v1 + (Nd * dt) ** 2 * a_range[1] / 2 + (
                    v1 - (Nd * dt) * a_range[1]) ** 2 / (2 * a_range[0]), name="headway3")
        model.addConstr(q[0] + q[1] + q[2] == 1, name="q_sum")

        model.addConstr(v1 == self.v_f + a0 * dt, name="v2_def")
        model.addConstr(p1 == self.p_f + self.v_f * dt + a0 * (dt ** 2) / 2, name="p2_def")
        model.addConstr(self.pd_l - p1 >= h1, name="safety_dis")

        # Objective function
        obj = QuadExpr()
        obj += w[0] * (v1 - self.vd_l) ** 2
        obj += w[1] * (p1 - self.pd_l) ** 2
        obj += w[2] * a0 ** 2

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        # self.debug(model)
        # sys.exit(0)

        a_opt = 0
        h_opt = 0
        if model.Status == GRB.OPTIMAL:
            a_opt = a0.X
            h_opt = h1.X

        return a_opt, h_opt

    def connected_strategy(self):
        model = Model("Connected model")
        model.setParam('OutputFlag', False)

        a = model.addVars(Np, lb=a_range[0], ub=a_range[1], name="a")
        v = model.addVars(Np, lb=v_range[0], ub=v_range[1], name="v")
        p = model.addVars(Np, lb=self.p_f, ub=GRB.INFINITY, name="p")
        h = model.addVar(lb=g_min, ub=GRB.INFINITY, name="h")
        am = model.addVar(lb=a_range[0], ub=0, name="am")
        q = model.addVars(3, vtype=GRB.BINARY, name="q")

        model.addConstr(M * (1 - q[0]) >= self.vh_l[Ns] - v[0] - (Ns * dt) * am + (Nd * dt) * a_range[1], name="q1")
        model.addQConstr(
            h + M * (1 - q[0]) >= -(v[0] + Ns * dt * am) ** 2 / (2 * a_range[0]) + (Nd + Ns) * dt * v[0] +
            (Nd + Ns / 2) * Ns * (dt ** 2) * am - self.ph_l[Ns] + self.ph_l[0], name="headway1")
        model.addConstr(M * (1 - q[1]) >= v[0] + Ns * dt * am - Nd * dt * a_range[1] - self.vh_l[Ns], name="q2_1")
        model.addConstr(M * (1 - q[1]) >= self.vh_l[Ns] - v[0] - Ns * dt * am, name="q2_2")
        model.addQConstr(
            h + M * (1 - q[1]) >= -(self.vh_l[Ns] + Nd * dt * a_range[1]) ** 2 / (2 * a_range[0]) + (Nd + Ns) * dt * v[
                0]
            - (self.vh_l[Ns] - v[0] - Ns * dt * am) ** 2 / (2 * a_range[1]) + (
                    Nd + Ns / 2) * Ns * dt ** 2 * am + 1 / 2 * (Nd * dt) ** 2 * a_range[1] - self.ph_l[Ns] +
            self.ph_l[0], name="headway2")
        model.addConstr(M * (1 - q[2]) >= v[0] - Ns * dt * (-am) - self.vh_l[Ns], name="q3")
        model.addQConstr(
            h + M * (1 - q[2]) >= -(v[0] + Ns * dt * am + Nd * dt * a_range[1]) ** 2 / (2 * a_range[0]) + (
                    Nd + Ns) * dt * v[0] +
            (Nd + Ns / 2) * Ns * dt ** 2 * am + 1 / 2 * (Nd * dt) ** 2 * a_range[1] - self.ph_l[Ns] + self.ph_l[0],
            name="headway3")
        model.addConstr(q[0] + q[1] + q[2] == 1, name="q_sum")

        model.addConstr(v[0] == self.v_f + a[0] * dt, name="v_0")
        model.addConstr(p[0] == self.p_f + v[0] * dt + a[0] * (dt ** 2) / 2, name="p_0")
        model.addConstr(self.ph_l[0] - p[0] >= h, name=f"safety_dis_{0}")
        for t in range(1, Np):
            model.addConstr(v[t] == v[t - 1] + a[t] * dt, name=f"v_{t}")
            model.addConstr(p[t] == p[t - 1] + v[t - 1] * dt + a[t] * (dt ** 2) / 2, name=f"p_{t}")
            model.addConstr(self.ph_l[t] - p[t] >= h, name=f"safety_dis_{t}")

        obj = QuadExpr()
        for t in range(Np):
            obj += w[0] * (self.vh_l[t] - v[t]) ** 2
            obj += w[1] * (self.ph_l[t] - p[t]) ** 2
            obj += w[2] * a[t] ** 2

        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        # self.debug(model)
        # sys.exit(0)

        a_opt = np.zeros(Np)
        h_opt = 0
        if model.Status == GRB.OPTIMAL:
            self.am = am.X
            for t in range(Np):
                a_opt[t] = a[t].X
                h_opt = h.X

        return a_opt[0], h_opt

    def protected_strategy(self):
        if self.protected_time > Ns + Nd + 1:
            model = Model("Connected model 2")
            model.setParam('OutputFlag', False)

            a = model.addVars(Np, lb=a_range[0], ub=a_range[1], name="a")
            v = model.addVars(Np, lb=v_range[0], ub=v_range[1], name="v")
            p = model.addVars(Np, lb=self.p_f, ub=GRB.INFINITY, name="p")
            h = model.addVar(lb=g_min, ub=GRB.INFINITY, name="h")
            am = model.addVar(lb=a_range[0], ub=0, name="am")
            q = model.addVars(3, vtype=GRB.BINARY, name="q")

            # 计算中间变量
            model.addConstr(M * (1 - q[0]) >= self.vh_l[Ns] - v[0] - (Ns * dt) * am + (Nd * dt) * a_range[1], name="q1")
            model.addQConstr(
                h + M * (1 - q[0]) >= -(v[0] + Ns * dt * am) ** 2 / (2 * a_range[0]) + Nd * dt * v[0] +
                Nd * Ns * (dt ** 2) * am, name="headway1")
            model.addConstr(M * (1 - q[1]) >= v[0] + Ns * dt * am - Nd * dt * a_range[1] - self.vh_l[Ns], name="q2_1")
            model.addConstr(M * (1 - q[1]) >= self.vh_l[Ns] - v[0] - Ns * dt * am, name="q2_2")
            model.addQConstr(
                h + M * (1 - q[1]) >= -(self.vh_l[Ns] + Nd * dt * a_range[1]) ** 2 / (2 * a_range[0]) + Nd * dt *
                v[0] - (self.vh_l[Ns] - v[0] - Ns * dt * am) ** 2 / (2 * a_range[1]) +
                Nd * Ns * dt ** 2 * am + 1 / 2 * (Nd * dt) ** 2 * a_range[1], name="headway2")
            model.addConstr(M * (1 - q[2]) >= v[0] - Ns * dt * (-am) - self.vh_l[Ns], name="q3")
            model.addQConstr(
                h + M * (1 - q[2]) >= -(v[0] + Ns * dt * am + Nd * dt * a_range[1]) ** 2 / (2 * a_range[0]) +
                Nd * dt * v[0] + Nd * Ns * dt ** 2 * am + 1 / 2 * (Nd * dt) ** 2 * a_range[1], name="headway3")
            model.addConstr(q[0] + q[1] + q[2] == 1, name="q_sum")

            model.addConstr(v[0] == self.v_f + a[0] * dt, name="v_0")
            model.addConstr(p[0] == self.p_f + v[0] * dt + a[0] * (dt ** 2) / 2, name="p_0")
            model.addConstr(self.ph_l[0] - p[0] >= h, name=f"safety_dis_{0}")
            for t in range(1, Np):
                model.addConstr(v[t] == v[t - 1] + a[t] * dt, name=f"v_{t}")
                model.addConstr(p[t] == p[t - 1] + v[t - 1] * dt + a[t] * (dt ** 2) / 2, name=f"p_{t}")
                model.addConstr(self.ph_l[t] - p[t] >= h, name=f"safety_dis_{t}")

            obj = QuadExpr()
            for t in range(Np):
                obj += w[0] * (self.vh_l[t] - v[t]) ** 2
                obj += w[1] * (self.ph_l[t] - p[t]) ** 2
                obj += w[2] * a[t] ** 2

            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            a_opt = 0
            h_opt = 0
            if model.Status == GRB.OPTIMAL:
                a_opt = a[0].X
                h_opt = h.X

        else:
            a_opt = self.am
            h_opt = self.get_non_connected_safety_headway()
        self.protected_time = self.protected_time + 1

        return a_opt, h_opt

    def get_non_connected_safety_headway(self):
        if self.vd_l <= self.v_f:
            return (Nd * dt) * self.v_f - self.v_f ** 2 / 2 / a_range[0]
        elif self.v_f < self.vd_l <= self.v_f + (Nd * dt) * a_range[1]:
            return (Nd * dt) * self.v_f - self.v_f ** 2 / 2 / a_range[0] - (self.vd_l ** 2 - self.v_f ** 2) / 2 / \
                a_range[1]
        else:
            return (Nd * dt) * self.v_f - self.v_f ** 2 / 2 / a_range[0] + a_range[1] * (Nd * dt) ** 2 / 2

    def no_security_strategy_connected(self):
        model = Model("Connected model 2")
        model.setParam('OutputFlag', False)

        a = model.addVars(Np, lb=a_range[0], ub=a_range[1], name="a")
        v = model.addVars(Np, lb=v_range[0], ub=v_range[1], name="v")
        p = model.addVars(Np, lb=self.p_f, ub=GRB.INFINITY, name="p")
        s = model.addVars(Np, lb=0, ub=GRB.INFINITY, name="p")
        h = 30

        model.addConstr(v[0] == self.v_f + a[0] * dt, name="v_0")
        model.addConstr(p[0] == self.p_f + v[0] * dt + a[0] * (dt ** 2) / 2, name="p_0")
        for t in range(1, Np):
            model.addConstr(v[t] == v[t - 1] + a[t] * dt, name=f"v_{t}")
            model.addConstr(p[t] == p[t - 1] + v[t - 1] * dt + a[t] * (dt ** 2) / 2, name=f"p_{t}")
            model.addConstr(s[t] >= h - (self.ph_l[t] - p[t]), name=f"safety_dis_{t}")

        obj = QuadExpr()
        for t in range(Np):
            obj += w[0] * (self.vh_l[t] - v[t]) ** 2
            obj += w[1] * a[t] ** 2
            if t > 1:
                obj += M * s[t]

        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        # self.debug(model)
        # sys.exit(0)

        a_opt = 0
        if model.Status == GRB.OPTIMAL:
            a_opt = a[0].X

        return a_opt, 0, 0, 0, 0

    def compare(self):
        mode = self.last_mode
        a, h = self.connected_strategy()
        # a, h = self.non_connected_strategy()
        return a, mode, self.am, self.protected_time, h
