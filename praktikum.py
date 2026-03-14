# ============================================
# SIMULASI TANGKI AIR (WATER TANK) - STUDI KASUS 2.1
# Versi Professional & Lengkap
# ============================================

import numpy as np
from scipy.integrate import solve_ivp
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

# ====================
# 1. KONFIGURASI & SETUP
# ====================

@dataclass
class TankConfig:
    """Konfigurasi parameter tangki air"""
    
    # Dimensi tangki (silinder)
    tank_radius: float = 1.0
    tank_height: float = 2.0
    
    # Parameter inlet (pengisian)
    inlet_diameter: float = 0.05
    inlet_velocity: float = 1.0
    
    # Parameter outlet (pengosongan)
    outlet_diameter: float = 0.04
    outlet_coefficient: float = 0.6
    
    # Parameter operasi
    initial_height: float = 0.0
    is_inlet_open: bool = True
    is_outlet_open: bool = False
    
    # Parameter fisik
    gravity: float = 9.81
    water_density: float = 1000.0
    
    # Parameter simulasi
    simulation_time: float = 300.0
    time_step: float = 1.0
    
    # Atribut turunan
    tank_area: float = field(init=False, default=None)
    inlet_area: float = field(init=False, default=None)
    outlet_area: float = field(init=False, default=None)
    tank_volume: float = field(init=False, default=None)
    
    def __post_init__(self):
        self.tank_area   = np.pi * self.tank_radius**2
        self.inlet_area  = np.pi * (self.inlet_diameter / 2)**2
        self.outlet_area = np.pi * (self.outlet_diameter / 2)**2
        self.tank_volume = self.tank_area * self.tank_height
    
    def copy(self):
        params = {k: v for k, v in self.__dict__.items()
                  if k not in ['tank_area', 'inlet_area', 'outlet_area', 'tank_volume']}
        return TankConfig(**params)
    
    def update_parameter(self, name: str, value):
        setattr(self, name, value)
        self.__post_init__()


# ====================
# 2. MODEL FISIKA
# ====================

class TankPhysicsModel:
    
    def __init__(self, config: TankConfig):
        self.config = config
    
    def calculate_inlet_flowrate(self) -> float:
        if not self.config.is_inlet_open:
            return 0.0
        return self.config.inlet_area * self.config.inlet_velocity
    
    def calculate_outlet_flowrate(self, h: float) -> float:
        if not self.config.is_outlet_open or h <= 0:
            return 0.0
        h = max(0, h)
        v = self.config.outlet_coefficient * np.sqrt(2 * self.config.gravity * h)
        return self.config.outlet_area * v
    
    def calculate_height_change_rate(self, h: float) -> float:
        h = max(0, min(h, self.config.tank_height))
        Q_in  = self.calculate_inlet_flowrate()
        Q_out = self.calculate_outlet_flowrate(h)
        return (Q_in - Q_out) / self.config.tank_area
    
    def steady_state_height(self) -> Optional[float]:
        """Hitung tinggi steady-state (Q_in = Q_out)"""
        if not self.config.is_inlet_open or not self.config.is_outlet_open:
            return None
        Q_in = self.calculate_inlet_flowrate()
        Cd   = self.config.outlet_coefficient
        A_out = self.config.outlet_area
        g    = self.config.gravity
        # Q_in = Cd * A_out * sqrt(2*g*h)  =>  h = (Q_in / (Cd*A_out))^2 / (2g)
        h_ss = (Q_in / (Cd * A_out))**2 / (2 * g)
        if h_ss > self.config.tank_height:
            return None  # Tidak ada steady state (tangki akan overflow)
        return h_ss
    
    def calculate_fill_time(self) -> Optional[float]:
        if not self.config.is_inlet_open:
            return None
        Q_in = self.calculate_inlet_flowrate()
        if Q_in <= 0:
            return None
        vol_needed = self.config.tank_volume - (self.config.initial_height * self.config.tank_area)
        if vol_needed <= 0:
            return 0.0
        return vol_needed / Q_in
    
    def calculate_empty_time(self) -> Optional[float]:
        """Estimasi waktu pengosongan secara analitik"""
        if not self.config.is_outlet_open or self.config.initial_height <= 0:
            return None
        Cd   = self.config.outlet_coefficient
        A_t  = self.config.tank_area
        A_out = self.config.outlet_area
        g    = self.config.gravity
        h0   = self.config.initial_height
        denom = Cd * A_out * np.sqrt(2 * g)
        if denom <= 0:
            return None
        return (2 * A_t * np.sqrt(h0)) / denom


# ====================
# 3. ODE SYSTEM
# ====================

class TankDifferentialEquations:
    
    def __init__(self, physics: TankPhysicsModel):
        self.physics = physics
        self.config  = physics.config
    
    def system_equation(self, t: float, y: np.ndarray) -> np.ndarray:
        h = y[0]
        dh_dt = self.physics.calculate_height_change_rate(h)
        if h >= self.config.tank_height and dh_dt > 0:
            dh_dt = 0
        if h <= 0 and dh_dt < 0:
            dh_dt = 0
        return np.array([dh_dt])
    
    def get_initial_conditions(self) -> np.ndarray:
        return np.array([self.config.initial_height])


# ====================
# 4. SIMULATOR
# ====================

class TankSimulator:
    
    def __init__(self, config: TankConfig):
        self.config    = config
        self.physics   = TankPhysicsModel(config)
        self.equations = TankDifferentialEquations(self.physics)
        
        self.time_history        = None
        self.height_history      = None
        self.volume_history      = None
        self.inlet_flow_history  = None
        self.outlet_flow_history = None
        self.results             = None
    
    def run_simulation(self) -> Dict:
        t_span = (0, self.config.simulation_time)
        t_eval = np.arange(0, self.config.simulation_time + self.config.time_step,
                           self.config.time_step)
        y0 = self.equations.get_initial_conditions()
        
        solution = solve_ivp(
            fun=self.equations.system_equation,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        self.time_history   = solution.t
        self.height_history = np.clip(solution.y[0], 0, self.config.tank_height)
        self.volume_history = self.height_history * self.config.tank_area
        
        self.inlet_flow_history  = np.array([self.physics.calculate_inlet_flowrate()
                                             for _ in self.height_history])
        self.outlet_flow_history = np.array([self.physics.calculate_outlet_flowrate(h)
                                             for h in self.height_history])
        
        self.results = self._calculate_metrics()
        return self.results
    
    def _calculate_metrics(self) -> Dict:
        fill_time   = None
        empty_time  = None
        steady_time = None
        
        if self.config.initial_height < self.config.tank_height and self.config.is_inlet_open:
            idx = np.where(self.height_history >= self.config.tank_height * 0.99)[0]
            if len(idx) > 0:
                fill_time = self.time_history[idx[0]]
        
        if self.config.initial_height > 0 and self.config.is_outlet_open:
            idx = np.where(self.height_history <= 0.01)[0]
            if len(idx) > 0:
                empty_time = self.time_history[idx[0]]
        
        if self.config.is_inlet_open and self.config.is_outlet_open and len(self.time_history) > 1:
            dh = np.diff(self.height_history) / np.diff(self.time_history)
            idx = np.where(np.abs(dh) < 0.0001)[0]
            if len(idx) > 0:
                steady_time = self.time_history[idx[0]]
        
        out_pos = self.outlet_flow_history[self.outlet_flow_history > 0]
        
        return {
            'time_to_full':         fill_time,
            'time_to_empty':        empty_time,
            'time_to_steady_state': steady_time,
            'initial_height':       self.config.initial_height,
            'final_height':         self.height_history[-1],
            'max_height':           np.max(self.height_history),
            'min_height':           np.min(self.height_history),
            'initial_volume':       self.config.initial_height * self.config.tank_area,
            'final_volume':         self.volume_history[-1],
            'max_volume':           np.max(self.volume_history),
            'min_volume':           np.min(self.volume_history),
            'tank_capacity':        self.config.tank_volume,
            'avg_inlet_flow':       np.mean(self.inlet_flow_history),
            'avg_outlet_flow':      np.mean(out_pos) if len(out_pos) > 0 else 0,
            'max_outlet_flow':      np.max(self.outlet_flow_history),
            'is_full':              self.height_history[-1] >= self.config.tank_height * 0.99,
            'is_empty':             self.height_history[-1] <= 0.01,
            'is_steady_state':      steady_time is not None,
        }


# ====================
# 5. VISUALISASI 2D TANK
# ====================

def draw_tank_2d(config: TankConfig, current_height: float):
    """Gambar tangki 2D dengan tinggi air aktual"""
    fill_ratio = current_height / config.tank_height if config.tank_height > 0 else 0
    fill_ratio = max(0, min(1, fill_ratio))
    
    fig = go.Figure()
    
    # Dinding tangki
    tank_w = 2.0
    fig.add_shape(type="rect", x0=-tank_w/2, y0=0, x1=tank_w/2, y1=config.tank_height,
                  line=dict(color="#334155", width=3), fillcolor="rgba(0,0,0,0)")
    
    # Air dalam tangki
    water_color = f"rgba(30,120,220,{0.3 + 0.5*fill_ratio})"
    if current_height > 0:
        fig.add_shape(type="rect", x0=-tank_w/2 + 0.05, y0=0,
                      x1=tank_w/2 - 0.05, y1=current_height,
                      fillcolor=water_color, line=dict(color="rgba(30,120,220,0.6)", width=1))
    
    # Garis permukaan air (bergelombang simulasi)
    if current_height > 0:
        x_wave = np.linspace(-tank_w/2 + 0.05, tank_w/2 - 0.05, 60)
        y_wave = current_height + 0.01 * np.sin(10 * x_wave)
        fig.add_trace(go.Scatter(x=x_wave, y=y_wave, mode='lines',
                                 line=dict(color='rgba(100,180,255,0.8)', width=2),
                                 showlegend=False, hoverinfo='skip'))
    
    # Pipa inlet (kanan atas)
    if config.is_inlet_open:
        fig.add_shape(type="rect", x0=tank_w/2, y0=config.tank_height * 0.85,
                      x1=tank_w/2 + 0.3, y1=config.tank_height * 0.85 + config.inlet_diameter,
                      fillcolor="#22c55e", line=dict(color="#15803d", width=1))
        fig.add_annotation(x=tank_w/2 + 0.15, y=config.tank_height * 0.9,
                           text="IN", showarrow=False, font=dict(size=9, color="white"))
    
    # Pipa outlet (kiri bawah)
    if config.is_outlet_open:
        fig.add_shape(type="rect", x0=-tank_w/2 - 0.3, y0=0.02,
                      x1=-tank_w/2, y1=0.02 + config.outlet_diameter,
                      fillcolor="#ef4444", line=dict(color="#b91c1c", width=1))
        fig.add_annotation(x=-tank_w/2 - 0.15, y=0.06,
                           text="OUT", showarrow=False, font=dict(size=9, color="white"))
    
    # Label tinggi
    fig.add_annotation(x=tank_w/2 + 0.1, y=current_height,
                       text=f"  h = {current_height:.2f} m",
                       showarrow=True, arrowhead=2, arrowsize=1,
                       arrowcolor="#1e40af", font=dict(size=11, color="#1e40af"),
                       ax=50, ay=0)
    
    # Label persentase
    fig.add_annotation(x=0, y=current_height / 2 if current_height > 0.2 else 0.5,
                       text=f"{fill_ratio*100:.0f}%",
                       showarrow=False, font=dict(size=16, color="white",
                       family="Arial Black"))
    
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=80, t=30, b=20),
        xaxis=dict(range=[-1.5, 1.8], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(range=[-0.1, config.tank_height + 0.3],
                   title="Tinggi (m)", showgrid=True, gridcolor="#e2e8f0"),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
        title=dict(text="Visualisasi Tangki", font=dict(size=14, color="#334155"))
    )
    return fig


# ====================
# 6. VISUALISASI PLOTLY
# ====================

class PlotlyTankVisualization:
    
    @staticmethod
    def plot_tank_profile(simulator: TankSimulator):
        fig = go.Figure()
        time   = simulator.time_history / 60.0
        height = simulator.height_history
        config = simulator.config
        h_ss   = simulator.physics.steady_state_height()
        
        # Area fill
        fig.add_trace(go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([height, np.zeros(len(height))]),
            fill='toself', fillcolor='rgba(30,100,220,0.08)',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(
            x=time, y=height, mode='lines', name='Tinggi Air',
            line=dict(color='#1d4ed8', width=3),
            hovertemplate='Waktu: %{x:.1f} menit<br>Tinggi: %{y:.3f} m<extra></extra>'))
        
        fig.add_hline(y=config.tank_height, line_dash="dash",
                      line_color="#dc2626", opacity=0.8,
                      annotation_text=f"H_max = {config.tank_height:.1f} m",
                      annotation_font=dict(color="#dc2626"))
        
        if h_ss is not None:
            fig.add_hline(y=h_ss, line_dash="dot", line_color="#16a34a", opacity=0.8,
                          annotation_text=f"H_ss = {h_ss:.2f} m (Steady State)",
                          annotation_font=dict(color="#16a34a"))
        
        # Waktu fill / empty markers
        results = simulator.results
        if results['time_to_full']:
            fig.add_vline(x=results['time_to_full']/60, line_dash="dot",
                          line_color="#f59e0b", opacity=0.7,
                          annotation_text=f"Penuh @ {results['time_to_full']/60:.1f} menit")
        if results['time_to_empty']:
            fig.add_vline(x=results['time_to_empty']/60, line_dash="dot",
                          line_color="#7c3aed", opacity=0.7,
                          annotation_text=f"Kosong @ {results['time_to_empty']/60:.1f} menit")
        
        fig.update_layout(
            title=dict(text='Profil Ketinggian Air terhadap Waktu',
                       font=dict(size=18, color="#1e3a5f")),
            xaxis_title="Waktu (menit)",
            yaxis_title="Ketinggian Air (m)",
            hovermode="x unified", height=450,
            template="plotly_white",
            legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#cbd5e1',
                        borderwidth=1))
        return fig
    
    @staticmethod
    def plot_flow_rates(simulator: TankSimulator):
        fig = go.Figure()
        time     = simulator.time_history / 60.0
        Q_in     = simulator.inlet_flow_history  * 1000
        Q_out    = simulator.outlet_flow_history * 1000
        net_flow = Q_in - Q_out
        
        fig.add_trace(go.Scatter(
            x=time, y=Q_in, mode='lines', name='Q_in (Debit Masuk)',
            line=dict(color='#16a34a', width=2.5),
            hovertemplate='t=%{x:.1f} menit<br>Q_in=%{y:.3f} L/s<extra></extra>'))
        
        fig.add_trace(go.Scatter(
            x=time, y=Q_out, mode='lines', name='Q_out (Debit Keluar)',
            line=dict(color='#dc2626', width=2.5),
            hovertemplate='t=%{x:.1f} menit<br>Q_out=%{y:.3f} L/s<extra></extra>'))
        
        fig.add_trace(go.Scatter(
            x=time, y=net_flow, mode='lines', name='Q_net = Q_in − Q_out',
            line=dict(color='#7c3aed', width=2, dash='dash'),
            hovertemplate='t=%{x:.1f} menit<br>Q_net=%{y:.3f} L/s<extra></extra>'))
        
        # Shading positif/negatif net flow
        fig.add_trace(go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([np.maximum(net_flow, 0), np.zeros(len(time))]),
            fill='toself', fillcolor='rgba(22,163,74,0.1)',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([np.minimum(net_flow, 0), np.zeros(len(time))]),
            fill='toself', fillcolor='rgba(220,38,38,0.1)',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'))
        
        fig.add_hline(y=0, line_color="black", opacity=0.3)
        
        fig.update_layout(
            title=dict(text='Debit Aliran Inlet & Outlet',
                       font=dict(size=18, color="#1e3a5f")),
            xaxis_title="Waktu (menit)",
            yaxis_title="Debit (L/s)",
            hovermode="x unified", height=450,
            template="plotly_white",
            legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#cbd5e1', borderwidth=1))
        return fig
    
    @staticmethod
    def plot_volume_profile(simulator: TankSimulator):
        fig = go.Figure()
        time    = simulator.time_history / 60.0
        vol_L   = simulator.volume_history * 1000
        cap_L   = simulator.config.tank_volume * 1000
        
        fig.add_trace(go.Scatter(
            x=time, y=vol_L, mode='lines', name='Volume Air',
            line=dict(color='#0891b2', width=3),
            fill='tozeroy', fillcolor='rgba(8,145,178,0.1)',
            hovertemplate='t=%{x:.1f} menit<br>V=%{y:.0f} L<extra></extra>'))
        
        fig.add_hline(y=cap_L, line_dash="dash", line_color="#dc2626", opacity=0.8,
                      annotation_text=f"Kapasitas Maks = {cap_L:.0f} L")
        fig.add_hline(y=cap_L * 0.8, line_dash="dot", line_color="#f59e0b", opacity=0.7,
                      annotation_text="80% Kapasitas")
        
        fig.update_layout(
            title=dict(text='Volume Air dalam Tangki',
                       font=dict(size=18, color="#1e3a5f")),
            xaxis_title="Waktu (menit)",
            yaxis_title="Volume (Liter)",
            hovermode="x unified", height=450,
            template="plotly_white")
        return fig
    
    @staticmethod
    def plot_dashboard(simulator: TankSimulator):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Profil Ketinggian Air', 'Debit Aliran',
                            'Volume Air', 'Diagram Fase (Q_in vs Q_out)'),
            vertical_spacing=0.18,
            horizontal_spacing=0.12)
        
        time   = simulator.time_history / 60.0
        height = simulator.height_history
        vol_L  = simulator.volume_history * 1000
        config = simulator.config
        
        # 1. Tinggi air
        fig.add_trace(go.Scatter(x=time, y=height, mode='lines', name='Tinggi Air',
                                 line=dict(color='#1d4ed8', width=2)),
                      row=1, col=1)
        fig.add_hline(y=config.tank_height, line_dash="dash", line_color="red",
                      opacity=0.5, row=1, col=1)
        
        # 2. Debit
        fig.add_trace(go.Scatter(x=time, y=simulator.inlet_flow_history * 1000,
                                 mode='lines', name='Q_in',
                                 line=dict(color='#16a34a', width=2)),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=time, y=simulator.outlet_flow_history * 1000,
                                 mode='lines', name='Q_out',
                                 line=dict(color='#dc2626', width=2)),
                      row=1, col=2)
        
        # 3. Volume
        fig.add_trace(go.Scatter(x=time, y=vol_L, mode='lines', name='Volume',
                                 line=dict(color='#7c3aed', width=2),
                                 fill='tozeroy', fillcolor='rgba(124,58,237,0.08)'),
                      row=2, col=1)
        fig.add_hline(y=config.tank_volume * 1000, line_dash="dash", line_color="red",
                      opacity=0.5, row=2, col=1)
        
        # 4. Fase operasi
        fig.add_trace(go.Scatter(
            x=simulator.inlet_flow_history * 1000,
            y=simulator.outlet_flow_history * 1000,
            mode='markers',
            marker=dict(size=6, color=time, colorscale='Plasma',
                        showscale=True, colorbar=dict(title="Menit", x=1.05)),
            text=[f"t={t:.1f} menit<br>h={h:.2f} m" for t, h in zip(time, height)],
            hovertemplate='%{text}<extra></extra>',
            name='Fase Operasi', showlegend=False),
            row=2, col=2)
        
        max_flow = max(np.max(simulator.inlet_flow_history),
                       np.max(simulator.outlet_flow_history)) * 1000 + 0.1
        fig.add_trace(go.Scatter(x=[0, max_flow], y=[0, max_flow], mode='lines',
                                 line=dict(color='red', dash='dash'), name='Q_in=Q_out',
                                 showlegend=True),
                      row=2, col=2)
        
        fig.update_layout(height=750, template="plotly_white",
                          showlegend=True, hovermode="closest",
                          title=dict(text="Dashboard Simulasi Tangki Air",
                                     font=dict(size=20, color="#1e3a5f")))
        fig.update_xaxes(title_text="Waktu (menit)", row=1, col=1)
        fig.update_xaxes(title_text="Waktu (menit)", row=1, col=2)
        fig.update_xaxes(title_text="Waktu (menit)", row=2, col=1)
        fig.update_xaxes(title_text="Q_in (L/s)", row=2, col=2)
        fig.update_yaxes(title_text="Tinggi (m)", row=1, col=1)
        fig.update_yaxes(title_text="Debit (L/s)", row=1, col=2)
        fig.update_yaxes(title_text="Volume (L)", row=2, col=1)
        fig.update_yaxes(title_text="Q_out (L/s)", row=2, col=2)
        return fig
    
    @staticmethod
    def plot_sensitivity(param_label: str, param_values: List, metrics_list: List[Dict],
                         metric_key: str, metric_label: str, unit: str = ""):
        fig = go.Figure()
        y_vals = []
        for m in metrics_list:
            val = m.get(metric_key)
            y_vals.append(val / 60 if val is not None and 'time' in metric_key else
                          (val * 1000 if val is not None and 'flow' in metric_key else
                           (val if val is not None else 0)))
        
        fig.add_trace(go.Scatter(
            x=param_values, y=y_vals, mode='lines+markers',
            line=dict(color='#1d4ed8', width=2.5),
            marker=dict(size=8, color='#1d4ed8',
                        line=dict(color='white', width=1.5)),
            hovertemplate=f'{param_label}: %{{x}}<br>{metric_label}: %{{y:.2f}} {unit}<extra></extra>'))
        
        fig.update_layout(
            title=dict(text=f'Sensitivitas: {metric_label} vs {param_label}',
                       font=dict(size=16, color="#1e3a5f")),
            xaxis_title=param_label,
            yaxis_title=f"{metric_label} ({unit})" if unit else metric_label,
            height=380, template="plotly_white")
        return fig
    
    @staticmethod
    def plot_optimal_tank(daily_need_L: float, peak_flow_Ls: float,
                          radii: np.ndarray, heights: np.ndarray):
        """Heatmap kapasitas vs kebutuhan (optimasi ukuran)"""
        Z = np.zeros((len(heights), len(radii)))
        for i, h in enumerate(heights):
            for j, r in enumerate(radii):
                cap = np.pi * r**2 * h * 1000
                Z[i, j] = cap
        
        fig = go.Figure(data=go.Heatmap(
            x=radii, y=heights, z=Z,
            colorscale='Blues',
            colorbar=dict(title="Kapasitas (L)"),
            hovertemplate='r=%{x:.2f} m<br>h=%{y:.2f} m<br>Kapasitas=%{z:.0f} L<extra></extra>'))
        
        # Garis kebutuhan harian (50% buffer)
        needed = daily_need_L * 0.5
        # Cari kombinasi r,h yang mendekati nilai needed
        contour_z = [[needed] * len(radii)] * len(heights)
        fig.add_trace(go.Contour(
            x=radii, y=heights, z=Z,
            contours=dict(coloring='none', showlabels=True,
                          start=needed, end=needed, size=1),
            line=dict(color='#dc2626', width=2),
            showscale=False, name='Minimum Kapasitas',
            hoverinfo='skip'))
        
        fig.update_layout(
            title=dict(text='Peta Kapasitas Tangki (Radius vs Tinggi)',
                       font=dict(size=16, color="#1e3a5f")),
            xaxis_title="Jari-jari Tangki (m)",
            yaxis_title="Tinggi Tangki (m)",
            height=420, template="plotly_white")
        return fig


# ====================
# 7. SIDEBAR
# ====================

def create_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/96/water-tank.png", width=64)
    st.sidebar.title("⚙️ Parameter Tangki")
    
    st.sidebar.subheader("🏗️ Dimensi Tangki")
    tank_radius = st.sidebar.slider("Jari-jari (m)", 0.5, 3.0, 1.0, 0.1)
    tank_height = st.sidebar.slider("Tinggi Tangki (m)", 1.0, 5.0, 2.0, 0.1)
    
    cap_L = np.pi * tank_radius**2 * tank_height * 1000
    st.sidebar.info(f"📦 Kapasitas: **{cap_L:.0f} Liter**")
    
    st.sidebar.subheader("🟢 Inlet (Pengisian)")
    inlet_d_cm = st.sidebar.slider("Diameter Pipa Inlet (cm)", 1.0, 15.0, 5.0, 0.5)
    inlet_v    = st.sidebar.slider("Kecepatan Aliran (m/s)", 0.1, 3.0, 1.0, 0.1)
    Q_in_Ls    = np.pi * (inlet_d_cm/200)**2 * inlet_v * 1000
    st.sidebar.caption(f"Q_in = {Q_in_Ls:.3f} L/s")
    
    st.sidebar.subheader("🔴 Outlet (Pengosongan)")
    outlet_d_cm = st.sidebar.slider("Diameter Pipa Outlet (cm)", 1.0, 15.0, 4.0, 0.5)
    outlet_Cd   = st.sidebar.slider("Koefisien Discharge (Cd)", 0.3, 1.0, 0.6, 0.05)
    
    st.sidebar.subheader("📌 Kondisi Awal & Mode")
    initial_h = st.sidebar.slider("Tinggi Air Awal (m)", 0.0, tank_height, 0.0, 0.1)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        is_inlet_open  = st.checkbox("✅ Inlet", value=True)
    with col2:
        is_outlet_open = st.checkbox("✅ Outlet", value=False)
    
    st.sidebar.subheader("⏱️ Simulasi")
    sim_time = st.sidebar.slider("Durasi Simulasi (detik)", 60, 7200, 300, 60)
    
    with st.sidebar.expander("🔬 Parameter Lanjutan"):
        gravity = st.number_input("Gravitasi (m/s²)", 9.0, 10.0, 9.81, 0.01)
    
    config = TankConfig(
        tank_radius=tank_radius,
        tank_height=tank_height,
        inlet_diameter=inlet_d_cm / 100,
        inlet_velocity=inlet_v,
        outlet_diameter=outlet_d_cm / 100,
        outlet_coefficient=outlet_Cd,
        initial_height=initial_h,
        is_inlet_open=is_inlet_open,
        is_outlet_open=is_outlet_open,
        gravity=gravity,
        simulation_time=float(sim_time)
    )
    
    with st.sidebar.expander("📖 Panduan Skenario"):
        st.markdown("""
**1. Pengisian saja:** ✅ Inlet | ❌ Outlet  
**2. Pengosongan saja:** ❌ Inlet | ✅ Outlet  
**3. Bersamaan:** ✅ Inlet | ✅ Outlet  
→ Jika Q_in > Q_out: tangki terisi  
→ Jika Q_in < Q_out: tangki terkuras  
→ Jika Q_in = Q_out: *steady state*
        """)
    
    return config


# ====================
# 8. DISPLAY METRICS
# ====================

def display_tank_results(sim: TankSimulator, results: Dict):
    cap_L    = sim.config.tank_volume * 1000
    final_L  = results['final_volume'] * 1000
    fill_pct = (results['final_height'] / sim.config.tank_height) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("💧 Tinggi Air Akhir",
                  f"{results['final_height']:.2f} m",
                  delta=f"{fill_pct:.1f}% penuh")
        st.metric("📦 Volume Saat Ini", f"{final_L:.0f} L")
    
    with c2:
        t_full = results['time_to_full']
        st.metric("⏱️ Waktu Mencapai Penuh",
                  f"{t_full/60:.1f} menit" if t_full else "Tidak tercapai")
        st.metric("🟢 Debit Masuk Rata-rata",
                  f"{results['avg_inlet_flow']*1000:.3f} L/s")
    
    with c3:
        t_empty = results['time_to_empty']
        st.metric("⏱️ Waktu Mencapai Kosong",
                  f"{t_empty/60:.1f} menit" if t_empty else "Tidak tercapai")
        st.metric("🔴 Debit Keluar Maks",
                  f"{results['max_outlet_flow']*1000:.3f} L/s")
    
    with c4:
        st.metric("🏗️ Kapasitas Total", f"{cap_L:.0f} L")
        status_map = {
            (True,  False): ("🟢 Penuh",    "green"),
            (False, True ): ("⚫ Kosong",   "gray"),
            (False, False): ("🔄 Proses",   "blue"),
        }
        status, _ = status_map.get(
            (results['is_full'], results['is_empty']), ("🔄 Proses", "blue"))
        if results['is_steady_state']:
            status = "⚖️ Steady State"
        st.metric("📊 Status Akhir", status)


# ====================
# 9. HALAMAN MATEMATIKA
# ====================

def show_math_model(config: TankConfig):
    """Tampilkan model matematika yang digunakan"""
    st.markdown("## 📐 Model Matematika")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Persamaan Keseimbangan Volume:**
$$\\frac{dV}{dt} = Q_{in} - Q_{out}$$

Karena $V = A_{tangki} \\cdot h$:
$$A_{tangki} \\cdot \\frac{dh}{dt} = Q_{in} - Q_{out}$$

$$\\boxed{\\frac{dh}{dt} = \\frac{Q_{in} - Q_{out}}{A_{tangki}}}$$
        """)
        
        st.markdown("""
**Debit Masuk (Inlet):**
$$Q_{in} = A_{inlet} \\cdot v_{inlet}$$

**Hukum Torricelli (Outlet):**
$$Q_{out} = C_d \\cdot A_{outlet} \\cdot \\sqrt{2gh}$$
        """)
    
    with col2:
        st.markdown("""
**Kondisi Steady State** ($\\frac{dh}{dt} = 0$):
$$Q_{in} = Q_{out}$$
$$A_{inlet} \\cdot v_{inlet} = C_d \\cdot A_{outlet} \\cdot \\sqrt{2g \\cdot h_{ss}}$$

$$\\boxed{h_{ss} = \\frac{1}{2g}\\left(\\frac{Q_{in}}{C_d \\cdot A_{outlet}}\\right)^2}$$
        """)
        
        h_ss = TankPhysicsModel(config).steady_state_height()
        Q_in = TankPhysicsModel(config).calculate_inlet_flowrate()
        
        st.markdown(f"""
**Nilai Parameter Saat Ini:**
| Parameter | Nilai |
|-----------|-------|
| $A_{{tangki}}$ | {config.tank_area:.4f} m² |
| $A_{{inlet}}$  | {config.inlet_area*1e4:.4f} cm² |
| $A_{{outlet}}$ | {config.outlet_area*1e4:.4f} cm² |
| $Q_{{in}}$     | {Q_in*1000:.3f} L/s |
| $C_d$          | {config.outlet_coefficient:.2f} |
| $h_{{ss}}$     | {f"{h_ss:.3f} m" if h_ss else "N/A (overflow)"} |
        """)


# ====================
# 10. ANALISIS SENSITIVITAS
# ====================

def run_sensitivity_tab(config: TankConfig):
    st.subheader("🔬 Analisis Sensitivitas Parameter")
    st.markdown("""
    Analisis ini menunjukkan bagaimana perubahan satu parameter mempengaruhi
    hasil simulasi (waktu pengisian, tinggi steady state, debit keluar maksimum).
    """)
    
    param_options = {
        "Kecepatan Inlet (m/s)":    ("inlet_velocity",   np.linspace(0.2, 3.0, 10)),
        "Diameter Outlet (cm)":     ("outlet_diameter",  np.linspace(0.01, 0.12, 10)),
        "Koefisien Discharge (Cd)": ("outlet_coefficient", np.linspace(0.3, 0.9, 7)),
        "Jari-jari Tangki (m)":     ("tank_radius",      np.linspace(0.5, 3.0, 8)),
    }
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected = st.selectbox("Parameter yang dianalisis:", list(param_options.keys()))
        metric_opt = st.selectbox("Metrik output:", [
            "Waktu Mencapai Penuh (menit)",
            "Waktu Mencapai Kosong (menit)",
            "Debit Keluar Maksimum (L/s)"
        ])
        run_btn = st.button("▶ Jalankan Analisis", type="primary")
    
    if run_btn:
        param_name, values = param_options[selected]
        
        metric_map = {
            "Waktu Mencapai Penuh (menit)":    ("time_to_full",    "Waktu Penuh",   "menit"),
            "Waktu Mencapai Kosong (menit)":   ("time_to_empty",   "Waktu Kosong",  "menit"),
            "Debit Keluar Maksimum (L/s)":     ("max_outlet_flow", "Q_out Maks",    "L/s"),
        }
        m_key, m_label, m_unit = metric_map[metric_opt]
        
        all_metrics = []
        display_vals = []
        
        progress = st.progress(0, text="Menjalankan simulasi sensitivitas...")
        
        for i, v in enumerate(values):
            cfg = config.copy()
            cfg.update_parameter(param_name, v)
            sim = TankSimulator(cfg)
            metrics = sim.run_simulation()
            all_metrics.append(metrics)
            
            # Nilai display (untuk Diameter Outlet: ubah ke cm)
            dv = v * 100 if param_name == "outlet_diameter" else v
            display_vals.append(round(dv, 4))
            progress.progress((i + 1) / len(values),
                              text=f"Simulasi {i+1}/{len(values)} selesai...")
        
        progress.empty()
        
        with col2:
            fig = PlotlyTankVisualization.plot_sensitivity(
                selected, display_vals, all_metrics, m_key, m_label, m_unit)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabel hasil
        rows = []
        for dv, m in zip(display_vals, all_metrics):
            val = m.get(m_key)
            if 'time' in m_key and val:
                val = val / 60
            elif 'flow' in m_key and val:
                val = val * 1000
            rows.append({"Parameter": round(dv, 3), m_label: round(val, 3) if val else "N/A"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ====================
# 11. OPTIMASI UKURAN
# ====================

def run_optimization_tab(config: TankConfig):
    st.subheader("🎯 Optimasi Ukuran Tangki")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Input Kebutuhan Air")
        daily_need = st.number_input("Kebutuhan harian (Liter)", 500, 50000, 5000, 500)
        num_people = st.number_input("Jumlah penghuni", 10, 1000, 100, 10)
        peak_hours = st.number_input("Durasi jam puncak (jam)", 1, 12, 4, 1)
        reserve_pct = st.slider("Buffer cadangan (%)", 10, 100, 50, 10)
        
        per_capita = daily_need / num_people
        hourly_avg = daily_need / 24
        peak_flow  = daily_need / (peak_hours * 3600)  # m³/s
        min_cap    = daily_need * (1 + reserve_pct / 100)
        
        st.info(f"""
**Analisis Kebutuhan:**
- Konsumsi per orang: **{per_capita:.0f} L/hari**
- Rata-rata per jam: **{hourly_avg:.0f} L/jam** ({hourly_avg/3600*1000:.2f} L/s)
- Debit jam puncak: **{peak_flow*1000:.2f} L/s**
- Kapasitas minimum (dengan buffer {reserve_pct}%): **{min_cap:.0f} L**
        """)
    
    with col2:
        st.markdown("#### 📐 Rekomendasi Dimensi")
        current_cap = config.tank_volume * 1000
        
        if current_cap < min_cap:
            st.error(f"⚠️ Kapasitas saat ini ({current_cap:.0f} L) **tidak mencukupi**!")
        else:
            st.success(f"✅ Kapasitas saat ini ({current_cap:.0f} L) mencukupi.")
        
        st.markdown("**Opsi Dimensi Optimal:**")
        
        heights_opt = [1.5, 2.0, 2.5, 3.0]
        rows = []
        for h_opt in heights_opt:
            r_opt = np.sqrt(min_cap / 1000 / (np.pi * h_opt))
            cap   = np.pi * r_opt**2 * h_opt * 1000
            rows.append({
                "Tinggi (m)": h_opt,
                "Radius (m)": round(r_opt, 2),
                "Diameter (m)": round(r_opt * 2, 2),
                "Kapasitas (L)": round(cap, 0)
            })
        
        df_opt = pd.DataFrame(rows)
        st.dataframe(df_opt, use_container_width=True, hide_index=True)
    
    # Heatmap kapasitas
    st.markdown("#### 🗺️ Peta Kapasitas Tangki")
    radii   = np.linspace(0.3, 3.5, 30)
    heights = np.linspace(0.5, 5.0, 30)
    fig_map = PlotlyTankVisualization.plot_optimal_tank(daily_need, peak_flow, radii, heights)
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Garis merah menunjukkan batas kapasitas minimum yang dibutuhkan.")


# ====================
# 12. SKENARIO KHUSUS
# ====================

def run_scenario_tab(config: TankConfig):
    st.subheader("🔄 Simulasi Skenario Khusus")
    
    scenario = st.radio(
        "Pilih skenario:",
        ["1️⃣  Pengisian Penuh (Kosong → Penuh)",
         "2️⃣  Pengosongan Penuh (Penuh → Kosong)",
         "3️⃣  Siklus Pengisian & Pengosongan Bersamaan",
         "4️⃣  Perbandingan Beberapa Ukuran Tangki"])
    
    if st.button("▶ Jalankan Skenario", type="primary"):
        
        if scenario.startswith("1️⃣"):
            cfg = config.copy()
            cfg.initial_height = 0.0
            cfg.is_inlet_open  = True
            cfg.is_outlet_open = False
            cfg.simulation_time = max(config.simulation_time,
                                      TankPhysicsModel(cfg).calculate_fill_time() * 1.1
                                      if TankPhysicsModel(cfg).calculate_fill_time() else 300)
            sim = TankSimulator(cfg)
            res = sim.run_simulation()
            
            t_full = res['time_to_full']
            if t_full:
                st.success(f"✅ Tangki penuh dalam **{t_full/60:.1f} menit**")
                st.info(f"Q_in = {res['avg_inlet_flow']*1000:.3f} L/s | "
                        f"Volume = {config.tank_volume*1000:.0f} L")
            else:
                st.warning("Tangki tidak mencapai penuh dalam waktu simulasi.")
            
            st.plotly_chart(PlotlyTankVisualization.plot_tank_profile(sim),
                            use_container_width=True)
        
        elif scenario.startswith("2️⃣"):
            cfg = config.copy()
            cfg.initial_height = config.tank_height
            cfg.is_inlet_open  = False
            cfg.is_outlet_open = True
            t_est = TankPhysicsModel(cfg).calculate_empty_time()
            cfg.simulation_time = max(config.simulation_time,
                                      t_est * 1.2 if t_est else 600)
            sim = TankSimulator(cfg)
            res = sim.run_simulation()
            
            t_empty = res['time_to_empty']
            if t_empty:
                st.success(f"✅ Tangki kosong dalam **{t_empty/60:.1f} menit**")
                st.info(f"Q_out maks = {res['max_outlet_flow']*1000:.3f} L/s "
                        f"(Torricelli, h = {config.tank_height:.1f} m)")
            else:
                st.warning("Tangki tidak mencapai kosong dalam waktu simulasi.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(PlotlyTankVisualization.plot_tank_profile(sim),
                                use_container_width=True)
            with col2:
                st.plotly_chart(PlotlyTankVisualization.plot_flow_rates(sim),
                                use_container_width=True)
        
        elif scenario.startswith("3️⃣"):
            cfg = config.copy()
            cfg.is_inlet_open  = True
            cfg.is_outlet_open = True
            sim = TankSimulator(cfg)
            res = sim.run_simulation()
            
            h_ss = sim.physics.steady_state_height()
            
            col1, col2 = st.columns(2)
            with col1:
                if h_ss:
                    st.success(f"⚖️ Steady State: h = **{h_ss:.3f} m** "
                               f"({h_ss/cfg.tank_height*100:.1f}% dari tinggi maks)")
                elif res['is_full']:
                    st.warning("⚠️ Q_in > Q_out maks: tangki akan overflow!")
                else:
                    st.info("ℹ️ Tangki akan terkuras bertahap (Q_out > Q_in).")
                
                if res['time_to_steady_state']:
                    st.info(f"Waktu mencapai steady: {res['time_to_steady_state']/60:.1f} menit")
            with col2:
                Q_in_Ls  = sim.physics.calculate_inlet_flowrate() * 1000
                Q_out_ss = (sim.physics.calculate_outlet_flowrate(h_ss) * 1000
                            if h_ss else 0)
                st.metric("Q_in", f"{Q_in_Ls:.3f} L/s")
                st.metric("Q_out (di steady state)", f"{Q_out_ss:.3f} L/s")
            
            st.plotly_chart(PlotlyTankVisualization.plot_dashboard(sim),
                            use_container_width=True)
        
        else:  # 4️⃣ Perbandingan
            st.markdown("**Membandingkan 4 ukuran tangki** dengan parameter inlet/outlet sama:")
            
            sizes = [(0.5, 1.5), (1.0, 2.0), (1.5, 2.5), (2.0, 3.0)]
            fig_comp = go.Figure()
            colors = ['#1d4ed8', '#16a34a', '#dc2626', '#7c3aed']
            
            for (r, h), color in zip(sizes, colors):
                cfg = config.copy()
                cfg.tank_radius = r
                cfg.tank_height = h
                cfg.initial_height = 0.0
                cfg.is_inlet_open  = True
                cfg.is_outlet_open = False
                cfg.simulation_time = 3600
                sim = TankSimulator(cfg)
                sim.run_simulation()
                cap_L = np.pi * r**2 * h * 1000
                
                fig_comp.add_trace(go.Scatter(
                    x=sim.time_history / 60,
                    y=sim.height_history / h * 100,
                    mode='lines',
                    name=f'r={r}m, H={h}m ({cap_L:.0f} L)',
                    line=dict(color=color, width=2.5)))
            
            fig_comp.update_layout(
                title="Perbandingan Tingkat Pengisian (%) Berbagai Ukuran Tangki",
                xaxis_title="Waktu (menit)",
                yaxis_title="Tingkat Terisi (%)",
                height=450, template="plotly_white")
            
            st.plotly_chart(fig_comp, use_container_width=True)


# ====================
# 13. MAIN APP
# ====================

def main():
    st.set_page_config(
        page_title="Simulasi Tangki Air",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 1rem; }
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    h1 { color: #1e3a5f !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #1565c0 60%, #0288d1 100%);
                padding: 24px 32px; border-radius: 16px; margin-bottom: 20px; color: white;">
        <h1 style="color: white !important; margin:0; font-size:2.2rem;">
            💧 Simulasi Kontinu Sistem Tangki Air
        </h1>
        <p style="margin:8px 0 0 0; opacity:0.9; font-size:1.05rem;">
            Studi Kasus 2.1 — Model dinamika distribusi air asrama berbasis persamaan diferensial
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    config = create_sidebar()
    
    # ─── Jalankan simulasi utama ───
    with st.spinner("⚙️ Menjalankan simulasi..."):
        sim     = TankSimulator(config)
        results = sim.run_simulation()
    
    # ─── Metrics ───
    st.success("✅ Simulasi selesai!")
    display_tank_results(sim, results)
    
    st.markdown("---")
    
    # ─── Visualisasi 2D tangki + info cepat ───
    col_tank, col_info = st.columns([1, 2])
    with col_tank:
        st.plotly_chart(draw_tank_2d(config, results['final_height']),
                        use_container_width=True)
    with col_info:
        h_ss = sim.physics.steady_state_height()
        Q_in_Ls  = sim.physics.calculate_inlet_flowrate() * 1000
        Q_out_ss = (sim.physics.calculate_outlet_flowrate(h_ss) * 1000 if h_ss else 0)
        
        mode_label = ("🔄 Pengisian & Pengosongan" if config.is_inlet_open and config.is_outlet_open
                      else "🟢 Pengisian Saja" if config.is_inlet_open
                      else "🔴 Pengosongan Saja" if config.is_outlet_open
                      else "⛔ Tidak Ada Aliran")
        
        st.markdown(f"""
#### 📋 Ringkasan Kondisi Operasi
| Parameter | Nilai |
|-----------|-------|
| Mode Operasi | **{mode_label}** |
| Tinggi Air Awal | {config.initial_height:.2f} m |
| Tinggi Air Akhir | {results['final_height']:.2f} m |
| Kapasitas Tangki | {config.tank_volume*1000:.0f} L |
| Q_in | {Q_in_Ls:.3f} L/s |
| Q_out Akhir | {sim.physics.calculate_outlet_flowrate(results['final_height'])*1000:.3f} L/s |
| Tinggi Steady State | {f"{h_ss:.3f} m" if h_ss else "N/A"} |
| Status | {"⚖️ Steady State" if results['is_steady_state'] else "🟢 Penuh" if results['is_full'] else "⚫ Kosong" if results['is_empty'] else "🔄 Proses"} |
        """)
    
    st.markdown("---")
    
    # ─── TABS UTAMA ───
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Ketinggian Air",
        "💧 Debit Aliran",
        "📦 Volume",
        "📊 Dashboard",
        "🔬 Sensitivitas",
        "🎯 Optimasi Ukuran",
        "🔄 Skenario Khusus"
    ])
    
    with tab1:
        st.subheader("Pertanyaan 1 & 3: Profil Ketinggian Air")
        st.caption("Berapa lama waktu pengisian? Bagaimana profil tinggi air terhadap waktu?")
        st.plotly_chart(PlotlyTankVisualization.plot_tank_profile(sim),
                        use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            t_fill = sim.physics.calculate_fill_time()
            if t_fill:
                st.info(f"⏱️ Estimasi waktu pengisian: **{t_fill/60:.1f} menit** "
                        f"({t_fill:.0f} detik)\n\nQ_in = {Q_in_Ls:.3f} L/s")
            else:
                st.info("Inlet tidak aktif.")
        with c2:
            t_empty = sim.physics.calculate_empty_time()
            if t_empty:
                st.info(f"⏱️ Estimasi waktu pengosongan: **{t_empty/60:.1f} menit** "
                        f"({t_empty:.0f} detik)\n\nFormula Torricelli: "
                        f"$t = \\frac{{2A_t \\sqrt{{h_0}}}}{{C_d A_{{out}} \\sqrt{{2g}}}}$")
            else:
                st.info("Outlet tidak aktif atau tangki kosong.")
    
    with tab2:
        st.subheader("Pertanyaan 2: Analisis Debit Aliran")
        st.caption("Berapa lama waktu pengosongan? Bagaimana perubahan debit outlet sesuai hukum Torricelli?")
        st.plotly_chart(PlotlyTankVisualization.plot_flow_rates(sim),
                        use_container_width=True)
        
        if config.is_inlet_open and config.is_outlet_open:
            t_ss = results['time_to_steady_state']
            if t_ss:
                st.success(f"⚖️ Kondisi steady state (Q_in = Q_out) tercapai pada **t = {t_ss/60:.1f} menit**")
            else:
                st.info("Steady state belum tercapai dalam durasi simulasi ini.")
    
    with tab3:
        st.subheader("Volume Air dalam Tangki")
        st.plotly_chart(PlotlyTankVisualization.plot_volume_profile(sim),
                        use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Volume Awal",  f"{results['initial_volume']*1000:.0f} L")
            st.metric("Volume Akhir", f"{results['final_volume']*1000:.0f} L")
        with c2:
            dv = (results['final_volume'] - results['initial_volume']) * 1000
            st.metric("Perubahan Volume", f"{'▲' if dv >= 0 else '▼'} {abs(dv):.0f} L")
            pct = results['final_volume'] / config.tank_volume * 100
            st.metric("Kapasitas Terpakai", f"{pct:.1f}%")
    
    with tab4:
        st.subheader("Dashboard Lengkap")
        st.plotly_chart(PlotlyTankVisualization.plot_dashboard(sim),
                        use_container_width=True)
    
    with tab5:
        run_sensitivity_tab(config)
    
    with tab6:
        st.subheader("Pertanyaan 5: Optimasi Ukuran Tangki")
        st.caption("Menentukan dimensi optimal berdasarkan kebutuhan air nyata.")
        run_optimization_tab(config)
    
    with tab7:
        st.subheader("Pertanyaan 3 & 4: Skenario Operasi")
        st.caption("Simulasi skenario pengisian, pengosongan, dan bersamaan.")
        run_scenario_tab(config)
    
    # ─── Model Matematika ───
    with st.expander("📐 Model Matematika & Persamaan"):
        show_math_model(config)
    
    # ─── Data lengkap & download ───
    with st.expander("📋 Data Simulasi & Export"):
        df = pd.DataFrame({
            'Waktu (s)':        sim.time_history,
            'Waktu (menit)':    sim.time_history / 60,
            'Tinggi Air (m)':   sim.height_history,
            'Volume (L)':       sim.volume_history * 1000,
            'Q_in (L/s)':       sim.inlet_flow_history  * 1000,
            'Q_out (L/s)':      sim.outlet_flow_history * 1000,
            'Q_net (L/s)':      (sim.inlet_flow_history - sim.outlet_flow_history) * 1000,
        })
        st.dataframe(df.style.format({
            'Waktu (s)': '{:.0f}', 'Waktu (menit)': '{:.2f}',
            'Tinggi Air (m)': '{:.4f}', 'Volume (L)': '{:.1f}',
            'Q_in (L/s)': '{:.4f}', 'Q_out (L/s)': '{:.4f}', 'Q_net (L/s)': '{:.4f}',
        }), use_container_width=True, height=300)
        
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv,
                           file_name="simulasi_tangki_air.csv", mime="text/csv")
    
    # Footer
    st.markdown("""
    <div style='text-align:center; padding: 20px; color: #64748b; font-size: 0.9rem;'>
        <b>Simulasi Tangki Air — Studi Kasus 2.1</b><br>
        Model ODE kontinu untuk sistem distribusi air asrama<br>
        Menggunakan metode RK45 (Runge-Kutta orde 4-5) | Hukum Torricelli untuk outlet
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()