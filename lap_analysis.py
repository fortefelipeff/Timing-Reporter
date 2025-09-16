#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lap_analysis.py (versão estável)
--------------------------------
- Relatório estático (matplotlib) e interativo (Plotly)
- Hover correto (piloto certo, tempo do ponto, Δ sessão)
- Tabela ordenável por clique (JS seguro, fora de f-strings)

Uso:
  python lap_analysis.py --csv "CARRERA - TREINO OFICIAL OPCIONAL 1 - laptimes.csv" --outdir "timing_report"
  python lap_analysis.py --csv "..." --outdir "timing_report" --interactive --invert-y

Dependências:
  pip install pandas matplotlib
  (interativo) pip install plotly
"""

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

# matplotlib (modo estático)
# ---------- JS/CSS (ordenar tabela) ----------
# Usamos string *raw* para não gerar SyntaxWarning com sequências como \d
SORT_JS = r"""
<script>
document.addEventListener('DOMContentLoaded', function () {
  const tbl = document.getElementById('summary-table');
  if (!tbl || !tbl.tHead) return;
  const headerCells = Array.from(tbl.tHead.rows[0].cells);

  const typeByHeader = {
    "Piloto": "text",
    "Voltas": "num",
    "Melhor Volta": "time",
    "Volta Média": "time",
    "Melhor S1": "time",
    "Melhor S2": "time",
    "Melhor S3": "time",
    "Δ p/ Líder": "time"
  };

  // prepara cabeçalhos (guarda rótulo base sem setas)
  headerCells.forEach((th, idx) => {
    const base = th.textContent.replace(/[▲▼]/g,'').trim();
    th.dataset.baseLabel = base;
    th.dataset.colType = typeByHeader[base] || "text";
    th.style.cursor = 'pointer';
    th.title = "Clique para ordenar";
    th.addEventListener('click', () => sortBy(idx, th.dataset.colType, th));
  });

  function resetHeaders() {
    headerCells.forEach(h => {
      h.dataset.sortDir = '';
      h.innerText = h.dataset.baseLabel;
    });
  }

  function parseTime(text) {
    text = (text || "").trim().replace(',', '.');
    if (!text) return NaN;
    const parts = text.split(':');
    if (parts.length === 1) return parseFloat(parts[0]);
    if (parts.length === 2) return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
    if (parts.length === 3) return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
    return NaN;
  }

  function keyFor(td, type) {
    const text = td ? td.textContent.trim() : "";
    if (type === 'num') {
      const v = parseFloat(text.replace(/[^\d\.\-]/g, ''));
      return isNaN(v) ? Infinity : v;
    }
    if (type === 'time') {
      const v = parseTime(text);
      return isNaN(v) ? Infinity : v;
    }
    return text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  }

  function sortBy(colIdx, type, th) {
    const tbody = tbl.tBodies[0];
    const rows = Array.from(tbody.rows);
    const current = th.dataset.sortDir === 'asc' ? 'asc' : th.dataset.sortDir === 'desc' ? 'desc' : null;
    const dir = current === 'asc' ? 'desc' : 'asc';

    rows.sort((a, b) => {
      const ka = keyFor(a.cells[colIdx], type);
      const kb = keyFor(b.cells[colIdx], type);
      if (ka === kb) return 0;
      if (dir === 'asc') return (ka > kb) ? 1 : -1;
      return (ka < kb) ? 1 : -1;
    });

    // reseta todos os cabeçalhos para o rótulo base e seta a seta apenas no clicado
    resetHeaders();
    th.dataset.sortDir = dir;
    th.innerText = th.dataset.baseLabel + (dir === 'asc' ? ' ▲' : ' ▼');

    rows.forEach(r => tbody.appendChild(r));
  }
});
</script>
<style>
#summary-table thead th { position: sticky; top:0; background:#6792AB; color:#fff; }
#summary-table thead th:hover { background:#557a92; }
</style>
"""
# ---------- Utilidades de tempo ----------
def _to_seconds(time_str):
    """Converte 'M:SS.mmm' (ou 'H:MM:SS.mmm') para segundos (float). Suporta vírgula como decimal."""
    if pd.isna(time_str):
        return np.nan
    s = str(time_str).strip().replace(',', '.')
    if not s or s.lower() in ('nan', 'none'):
        return np.nan
    parts = s.split(':')
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            m = float(parts[0]); sec = float(parts[1])
            return m*60 + sec
        elif len(parts) == 3:
            h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
            return h*3600 + m*60 + sec
    except ValueError:
        return np.nan
    return np.nan

def _fmt_mmss(x):
    if pd.isna(x):
        return ""
    x = float(x)
    m, s = divmod(x, 60.0)
    return f"{int(m)}:{s:06.3f}"

# ---------- Categoria / Cores ----------
def _extract_category(name: str) -> str:
    s = (name or "").upper()
    if "CAR R" in s:
        return "CAR R"
    if "CAR S" in s:
        return "CAR S"
    if " CAR" in s or s.endswith("CAR"):
        return "CAR"
    return "OUTROS"

def _driver_colors(drivers: list[str]) -> dict:
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd","#e6550d","#31a354","#756bb1","#636363",
        "#9ecae1","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
        "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf","#999999"
    ]
    m = {}
    for i, d in enumerate(drivers):
        m[str(d)] = palette[i % len(palette)]
    return m

def _contrast_color(hex_color: str) -> str:
    """Retorna '#000' ou '#fff' conforme contraste com a cor de fundo."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return '#000'
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # luminância relativa simples
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return '#000' if lum > 0.6 else '#fff'

def _prepare_plotly_dataset(lap_df: pd.DataFrame,
                            max_lap_sec: float,
                            max_sector_sec: float,
                            hide_out_in: bool) -> tuple[pd.DataFrame, list, dict[str, str]]:
    """Filtra dados e retorna DataFrame ordenado, ordem de pilotos e mapa de cores."""
    df = lap_df.copy()
    if "Lap Tm_sec" in df.columns:
        df = df[df["Lap Tm_sec"].notna() & (df["Lap Tm_sec"] < max_lap_sec)]
    for col in ["S1 Tm_sec", "S2 Tm_sec", "S3 Tm_sec"]:
        if col in df.columns:
            df = df[df[col].isna() | (df[col] < max_sector_sec)]
    if hide_out_in and "LapType" in df.columns:
        df = df[(df["LapType"] == "normal") | (df["LapType"].isna())]

    if "Driver" in df.columns:
        order_idx = (df.groupby("Driver")["Lap Tm_sec"].min()
                       .sort_values()
                       .index.tolist())
        df["Driver"] = pd.Categorical(df["Driver"], categories=order_idx, ordered=True)
        df["Category"] = df["Driver"].astype(str).apply(_extract_category)
    else:
        order_idx = []
        df["Category"] = ""
    color_map = _driver_colors([str(d) for d in order_idx])
    return df, order_idx, color_map

def _make_line_figure(go_module,
                      df: pd.DataFrame,
                      order: list,
                      color_map: dict[str, str],
                      ycol: str,
                      title: str,
                      invert_y: bool,
                      interactive: bool = False) -> "go.Figure | None":
    if ycol not in df.columns:
        return None
    series = df[ycol].dropna()
    if series.empty:
        return None
    session_best = series.min()
    fig = go_module.Figure()
    for drv in order:
        sub = df[df["Driver"] == drv].sort_values("Lap")
        sub = sub[sub[ycol].notna()]
        if sub.empty:
            continue
        mmss = sub[ycol].apply(_fmt_mmss).astype(str).to_numpy()
        custom = np.stack([
            sub["Driver"].astype(str).to_numpy(),
            mmss,
            (sub[ycol] - session_best).to_numpy(),
        ], axis=-1)
        color = color_map.get(str(drv))
        line_color = color if color is not None else "#333"
        fig.add_trace(go_module.Scatter(
            x=sub["Lap"],
            y=sub[ycol],
            mode="lines+markers",
            name=str(drv),
            customdata=custom,
            line=dict(color=line_color),
            marker=dict(color=line_color, size=6),
            hovertemplate=(
                "Piloto: %{customdata[0]}<br>Volta: %{x}<br>"
                "Tempo: %{y:.3f}s (%{customdata[1]})<br>"
                "Δ sessão: %{customdata[2]:.3f}s<extra></extra>"
            )
        ))
    if not fig.data:
        return None
    fig.update_layout(
        title=title,
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=70, r=260, t=24, b=60),
    )
    xaxis_kwargs = dict(title="Volta", tickangle=-45, tickmode="linear", dtick=1,
                        showgrid=True, griddash="dot")
    yaxis_kwargs = dict(title="Tempo (s)", showgrid=True, griddash="dot")
    if interactive:
        fig.update_layout(hovermode="closest", hoverdistance=8, spikedistance=8)
        xaxis_kwargs.update(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
        yaxis_kwargs.update(showspikes=True, spikemode="across", spikethickness=1)
    fig.update_xaxes(**xaxis_kwargs)
    fig.update_yaxes(**yaxis_kwargs)
    if invert_y:
        fig.update_yaxes(autorange="reversed")
    return fig

def _make_box_figure(go_module,
                     df: pd.DataFrame,
                     order: list,
                     color_map: dict[str, str],
                     invert_y: bool) -> "go.Figure | None":
    if "Lap Tm_sec" not in df.columns:
        return None
    fig = go_module.Figure()
    has_data = False
    for drv in order:
        sub = df[df["Driver"] == drv]["Lap Tm_sec"].dropna()
        if sub.empty:
            continue
        has_data = True
        color = color_map.get(str(drv), "#333")
        fig.add_trace(go_module.Box(
            y=sub,
            name=str(drv),
            boxpoints="outliers",
            marker=dict(color=color, size=3),
            line=dict(color=color)
        ))
    if not has_data:
        return None
    fig.update_layout(title="Distribuição de Tempos por Piloto",
                      margin=dict(l=70, r=60, t=40, b=120), showlegend=False)
    fig.update_yaxes(title="Tempo (s)", showgrid=True, griddash="dot")
    if invert_y:
        fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-45)
    return fig

def _make_scatter_figure(go_module,
                         df: pd.DataFrame,
                         order: list,
                         color_map: dict[str, str]) -> "go.Figure | None":
    if not {"S1 Tm_sec", "S2 Tm_sec"}.issubset(df.columns):
        return None
    fig = go_module.Figure()
    has_data = False
    for drv in order:
        cols = ["S1 Tm_sec", "S2 Tm_sec", "Lap", "Lap Tm_sec"]
        if not set(cols).issubset(df.columns):
            continue
        sub = df[df["Driver"] == drv][cols].dropna()
        if sub.empty:
            continue
        has_data = True
        color = color_map.get(str(drv), "#333")
        fig.add_trace(go_module.Scatter(
            x=sub["S1 Tm_sec"],
            y=sub["S2 Tm_sec"],
            mode="markers",
            name=str(drv),
            marker=dict(color=color, size=5, opacity=0.8),
            text=[f"Volta {int(l)} — {_fmt_mmss(t)}" for l, t in zip(sub["Lap"], sub["Lap Tm_sec"])],
            hovertemplate="Piloto: %{fullData.name}<br>S1: %{x:.3f}s<br>S2: %{y:.3f}s<br>%{text}<extra></extra>"
        ))
    if not has_data:
        return None
    fig.update_layout(title="Dispersão S1 × S2",
                      xaxis_title="S1 (s)", yaxis_title="S2 (s)",
                      margin=dict(l=70, r=60, t=40, b=60))
    return fig

def _make_heatmap(go_module,
                  df: pd.DataFrame,
                  order: list) -> "go.Figure | None":
    if "Lap Tm_sec" not in df.columns or not order:
        return None
    laps = df.get("Lap")
    if laps is None:
        return None
    laps_all = sorted(pd.Series(laps).dropna().unique().astype(int).tolist())
    if not laps_all:
        return None
    session_best = df["Lap Tm_sec"].min()
    if pd.isna(session_best):
        return None
    matrix = []
    has_values = False
    for drv in order:
        sub = df[df["Driver"] == drv].set_index("Lap")["Lap Tm_sec"]
        row = []
        for lap in laps_all:
            value = sub.get(lap, np.nan)
            if pd.isna(value):
                row.append(np.nan)
            else:
                has_values = True
                row.append(float(value - session_best))
        matrix.append(row)
    if not has_values:
        return None
    fig = go_module.Figure(data=go_module.Heatmap(
        z=matrix,
        x=laps_all,
        y=[str(d) for d in order],
        colorscale="RdYlGn_r",
        colorbar=dict(title="Δ sess (s)")
    ))
    fig.update_layout(title="Heatmap Δ p/ Melhor Sessão",
                      xaxis_title="Volta", yaxis_title="Piloto",
                      margin=dict(l=120, r=60, t=40, b=60))
    return fig

def _build_plotly_sections(go_module,
                           df: pd.DataFrame,
                           order: list,
                           color_map: dict[str, str],
                           invert_y: bool,
                           interactive: bool = False) -> list[tuple[str, "go.Figure"]]:
    sections: list[tuple[str, "go.Figure"]] = []
    total = _make_line_figure(go_module, df, order, color_map,
                              "Lap Tm_sec", "Tempo de Volta – Todos os Pilotos", invert_y,
                              interactive=interactive)
    if total is not None:
        sections.append(("Tempo de Volta – Todos os Pilotos", total))
    for col, title in [("S1 Tm_sec", "Setor 1"),
                       ("S2 Tm_sec", "Setor 2"),
                       ("S3 Tm_sec", "Setor 3")]:
        fig = _make_line_figure(go_module, df, order, color_map, col, title, invert_y,
                                interactive=interactive)
        if fig is not None:
            sections.append((title, fig))
    box = _make_box_figure(go_module, df, order, color_map, invert_y)
    if box is not None:
        sections.append(("Distribuição de Tempos por Piloto", box))
    heatmap = _make_heatmap(go_module, df, order)
    if heatmap is not None:
        sections.append(("Heatmap Δ p/ Melhor Sessão", heatmap))
    scatter = _make_scatter_figure(go_module, df, order, color_map)
    if scatter is not None:
        sections.append(("Dispersão S1 × S2", scatter))
    return sections

def _render_summary_table_html(summary_df: pd.DataFrame,
                               color_map: dict[str, str]) -> str:
    header_html = ''.join(f'<th>{html.escape(str(col))}</th>' for col in summary_df.columns)
    body_rows = []
    for _, row in summary_df.iterrows():
        piloto = str(row.get('Piloto', ''))
        bg = color_map.get(piloto, '#fff')
        fg = _contrast_color(bg)
        cells = ''.join(
            f'<td>{html.escape(str(row[col]))}</td>'
            for col in summary_df.columns
        )
        body_rows.append(f'<tr style="background:{bg};color:{fg}">{cells}</tr>')
    return (
        '<table id="summary-table" class="summary">'
        '<thead><tr>' + header_html + '</tr></thead>'
        '<tbody>' + ''.join(body_rows) + '</tbody></table>'
    )

def _build_summary_table_figure(go_module,
                                summary_df: pd.DataFrame,
                                color_map: dict[str, str]):
    header_vals = list(summary_df.columns)
    cell_vals = [summary_df[col].astype(str).tolist() for col in header_vals]
    pilotos = summary_df['Piloto'].astype(str).tolist() if 'Piloto' in summary_df else []
    row_colors = [color_map.get(p, '#fff') for p in pilotos]
    font_colors = [_contrast_color(c) for c in row_colors]
    fill_matrix = [row_colors[:] for _ in header_vals]
    font_matrix = [font_colors[:] for _ in header_vals]
    table_height = 120 + 28 * (len(summary_df) + 1)
    fig = go_module.Figure(data=[go_module.Table(
        header=dict(values=header_vals,
                    fill_color="#6792AB",
                    font=dict(color="white", size=12),
                    align="center"),
        cells=dict(values=cell_vals,
                   align="center",
                   fill_color=fill_matrix,
                   font=dict(color=font_matrix))
    )])
    fig.update_layout(title="Sumário",
                      margin=dict(l=20, r=20, t=40, b=20),
                      height=table_height)
    return fig

# ---------- Sumário padronizado ----------
def build_summary_df(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Cria DataFrame de sumário padronizado para HTML e PDF com colunas:
    Piloto, Melhor Volta, Volta Teorica, S1, Melhor S1, S2, Melhor S2, S3, Melhor S3, Melhor 3 Voltas, Melhor 5 Voltas
    """
    base = (df.groupby("Driver")
              .agg(Voltas=("Lap","nunique"),
                   Best=("Lap Tm_sec","min"),
                   BestS1=("S1 Tm_sec","min"),
                   BestS2=("S2 Tm_sec","min"),
                   BestS3=("S3 Tm_sec","min"))
              .reindex(order).reset_index())
    # Volta teórica
    base["Theo"] = base[["BestS1","BestS2","BestS3"]].sum(axis=1, min_count=1)
    # Setores da melhor volta do piloto
    try:
        idx_best = df.groupby("Driver")["Lap Tm_sec"].idxmin()
        best_secs = df.loc[idx_best, ["Driver","S1 Tm_sec","S2 Tm_sec","S3 Tm_sec"]]
        best_secs = best_secs.rename(columns={"S1 Tm_sec":"LapS1","S2 Tm_sec":"LapS2","S3 Tm_sec":"LapS3"})
        base = base.merge(best_secs, on="Driver", how="left")
    except Exception:
        base["LapS1"] = np.nan; base["LapS2"] = np.nan; base["LapS3"] = np.nan
    # Melhor 3/5 voltas
    def _best_seq(series: pd.Series, w: int) -> float:
        s = series.dropna().astype(float)
        if len(s) < w: return np.nan
        return s.rolling(window=w).mean().min()
    b3_list, b5_list = [], []
    for drv in order:
        s = df.loc[df["Driver"]==drv, "Lap Tm_sec"].dropna().astype(float)
        b3_list.append(np.nan if len(s)<3 else _best_seq(s,3))
        b5_list.append(np.nan if len(s)<5 else _best_seq(s,5))
    base["Best3"], base["Best5"] = b3_list, b5_list
    # Formatação
    for c in ["Best","Theo","LapS1","BestS1","LapS2","BestS2","LapS3","BestS3","Best3","Best5"]:
        base[c] = base[c].apply(_fmt_mmss)
    # Renomeia e reordena
    out = (base.rename(columns={
        "Driver":"Piloto",
        "Best":"Melhor Volta",
        "Theo":"Volta Teorica",
        "LapS1":"S1","BestS1":"Melhor S1",
        "LapS2":"S2","BestS2":"Melhor S2",
        "LapS3":"S3","BestS3":"Melhor S3",
        "Best3":"Melhor 3 Voltas","Best5":"Melhor 5 Voltas"
    })[["Piloto","Voltas","Melhor Volta","Volta Teorica","S1","Melhor S1","S2","Melhor S2","S3","Melhor S3","Melhor 3 Voltas","Melhor 5 Voltas"]])
    return out

# ---------- Stints / Pit detection ----------
def annotate_stints(df: pd.DataFrame, pit_gap_sec: float = 25.0) -> pd.DataFrame:
    """Anota Stint e LapType (normal/in/out) por piloto via limiar de gap.

    Marca volta como 'in' se tempo >= mediana_do_piloto + pit_gap_sec; a seguinte vira 'out'.
    """
    if df.empty:
        out = df.copy()
        out["Stint"], out["LapType"] = 1, "normal"
        return out
    out = df.sort_values(["Driver", "Lap"]).copy()
    medians = out.groupby("Driver")["Lap Tm_sec"].transform("median")
    gap = (
        pd.notna(out["Lap Tm_sec"]) & pd.notna(medians) &
        (out["Lap Tm_sec"] >= medians + float(pit_gap_sec))
    )
    out["LapType"] = "normal"
    out.loc[gap, "LapType"] = "in"
    out.loc[gap.groupby(out["Driver"]).shift(1, fill_value=False), "LapType"] = "out"
    stint_cum = gap.groupby(out["Driver"]).cumsum()
    out["Stint"] = stint_cum + 1
    out.loc[gap, "Stint"] = stint_cum.loc[gap]
    out["Stint"] = out["Stint"].astype("Int64")
    return out

# ---------- Parsing do CSV ----------
def parse_lap_data(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw["Driver"] = raw["Time of Day"].where(raw["Lap"].isna()).ffill()
    df = raw[raw["Lap"].notna()].copy()
    df["Lap"] = df["Lap"].astype(int)
    for col in ["Lap Tm", "S1 Tm", "S2 Tm", "S3 Tm"]:
        if col in df.columns:
            df[col + "_sec"] = df[col].map(_to_seconds)
        else:
            df[col + "_sec"] = np.nan
    cols = ["Driver", "Lap", "Lap Tm_sec", "S1 Tm_sec", "S2 Tm_sec", "S3 Tm_sec"]
    df = df[cols].sort_values(["Driver", "Lap"]).reset_index(drop=True)
    return df

# ---------- Métricas ----------
def compute_driver_metrics(df: pd.DataFrame):
    g = df.groupby("Driver", as_index=False)
    out = g.agg(
        Laps=("Lap", "nunique"),
        Best=("Lap Tm_sec", "min"),
        Avg=("Lap Tm_sec", "mean"),
        BestS1=("S1 Tm_sec", "min"),
        BestS2=("S2 Tm_sec", "min"),
        BestS3=("S3 Tm_sec", "min"),
    ).sort_values("Best").reset_index(drop=True)
    session_best = out["Best"].min()
    out["DiffFastest"] = out["Best"] - session_best
    show = out.copy()
    for c in [col for col in ["Best","Avg","BestS1","BestS2","BestS3","DiffFastest"] if col in show.columns]:
        show[c] = show[c].apply(_fmt_mmss)
    show = show.rename(columns={
        "Driver":"Piloto","Laps":"Voltas","Best":"Melhor Volta","Avg":"Volta Média",
        "BestS1":"Melhor S1","BestS2":"Melhor S2","BestS3":"Melhor S3","DiffFastest":"Δ p/ Líder"
    })
    # Reordena/remapeia para remover Volta Média e incluir Volta Teorica
    # removido: tabela não usada aqui; definido no fluxo de exportação
    # table_df = summary[["Driver","Voltas","Best","Theo","BestS1","BestS2","BestS3"]].rename(columns={
    #     "Driver":"Piloto","Voltas":"Voltas","Best":"Melhor Volta","Theo":"Volta Teorica",
    #     "BestS1":"Melhor S1","BestS2":"Melhor S2","BestS3":"Melhor S3"
    # })
    return out, show

# ---------- Estático (matplotlib) ----------
def _palette_for(drivers):
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(drivers))]
    return dict(zip(drivers, colors))

def plot_all(df: pd.DataFrame, outdir: Path, invert_y=False,
             max_lap_sec=200.0, max_sector_sec=100.0) -> dict:
    global plt
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib não está instalado. Rode: pip install matplotlib") from exc
    outdir.mkdir(parents=True, exist_ok=True)
    dff = df[df["Lap Tm_sec"].notna() & (df["Lap Tm_sec"] < max_lap_sec)].copy()
    for col in ["S1 Tm_sec","S2 Tm_sec","S3 Tm_sec"]:
        dff = dff[dff[col].isna() | (dff[col] < max_sector_sec)]
    drivers = dff["Driver"].dropna().unique().tolist()
    colors = _palette_for(drivers)

    def _styled_line(data, ycol, title, fname):
        fig, ax = plt.subplots(figsize=(14, 7.5), dpi=120)
        for drv in drivers:
            dd = data[data["Driver"] == drv]
            if dd.empty: continue
            ax.plot(dd["Lap"], dd[ycol], marker="o", markersize=3,
                    linewidth=1.4, label=drv, color=colors[drv])
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Volta", fontsize=11)
        ax.set_ylabel("Tempo (s)", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
        ax.tick_params(axis="x", labelrotation=45, labelsize=9)
        ax.tick_params(axis="y", labelsize=10)
        if invert_y: ax.invert_yaxis()
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, ncol=1)
        plt.tight_layout(rect=[0,0,0.78,1])
        path = outdir / fname
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    return {
        "lap": _styled_line(dff, "Lap Tm_sec", "Tempo de Volta – Todos os Pilotos", "01_lap_times.png"),
        "s1":  _styled_line(dff, "S1 Tm_sec",  "Setor 1 – Todos os Pilotos",      "02_sector1.png"),
        "s2":  _styled_line(dff, "S2 Tm_sec",  "Setor 2 – Todos os Pilotos",      "03_sector2.png"),
        "s3":  _styled_line(dff, "S3 Tm_sec",  "Setor 3 – Todos os Pilotos",      "04_sector3.png"),
    }

def build_html(report_dir: Path, img_paths: dict, summary_df_show: pd.DataFrame):
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#fafafa; }
    .container { max-width: 1280px; margin: 24px auto; }
    .box { background:#fff; border:1px solid #ddd; padding:16px; margin-bottom:24px; }
    h1 { margin: 0 0 8px 0; font-size: 22px; }
    h2 { margin: 0 0 8px 0; font-size: 18px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: center; }
    thead th { background:#3e6d8f; color:#fff; position: sticky; top: 0; }
    tr:nth-child(even){ background:#f6f6f6; }
    img { width: 100%; height: auto; border:1px solid #eee; }
    .meta { color:#666; font-size:12px; }
    """
    table_html = summary_df_show.to_html(index=False, border=0).replace('<table ', '<table id="summary-table" ')
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Timing Report</title>
<style>{css}</style>
</head>
<body>
  <div class="container">
    <div class="box">
      <h1>Tempo de Volta – Todos os Pilotos</h1>
      <div class="meta">Gerado em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    <div class="box"><h2>Tempo de Volta – Todos os Pilotos</h2><img src="{img_paths['lap'].name}" alt="Lap Times"></div>
    <div class="box"><h2>Setor 1</h2><img src="{img_paths['s1'].name}" alt="S1"></div>
    <div class="box"><h2>Setor 2</h2><img src="{img_paths['s2'].name}" alt="S2"></div>
    <div class="box"><h2>Setor 3</h2><img src="{img_paths['s3'].name}" alt="S3"></div>
    <div class="box"><h2>Sumário</h2>{table_html}<div><small>Clique nos cabeçalhos para ordenar ↑↓</small></div></div>
  </div>
</body>
</html>
"""
    out_path = report_dir / "index.html"
    html = html + SORT_JS  # anexa JS/CSS seguro
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

# ---------- Interativo (Plotly) ----------

    
def generate_report_interactive(lap_df: pd.DataFrame, out_html: str = "report.html",
                                max_lap_sec=200.0, max_sector_sec=100.0,
                                invert_y=False, hide_out_in: bool = False):
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as e:
        raise RuntimeError("Plotly não está instalado. Rode: pip install plotly") from e

    df, order, color_map = _prepare_plotly_dataset(
        lap_df,
        max_lap_sec=max_lap_sec,
        max_sector_sec=max_sector_sec,
        hide_out_in=hide_out_in,
    )
    summary_df = build_summary_df(df, order)
    fig_sections = _build_plotly_sections(go, df, order, color_map, invert_y, interactive=True)
    table_html = _render_summary_table_html(summary_df, color_map)

    chart_blocks: list[str] = []
    for idx, (title, figure) in enumerate(fig_sections):
        include_js = "inline" if idx == 0 else False
        snippet = pio.to_html(figure, include_plotlyjs=include_js, full_html=False)
        chart_blocks.append(f'    <div class="box"><h2>{title}</h2>{snippet}</div>')
    charts_html = "\n".join(chart_blocks)

    html = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Timing Report (Interativo)</title>
<style>
 body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
 .container {{ max-width: 1280px; margin: 24px auto; }}
 .box {{ background:#fff; border:1px solid #ddd; padding:16px; margin-bottom:24px; }}
 .summary {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
 .summary th, .summary td {{ border:1px solid #ddd; padding:6px 8px; text-align:center; }}
 .summary thead th {{ background:#6792AB; color:#fff; position: sticky; top: 0; }}
 .summary tr:nth-child(even){{ background:#f6f6f6; }}
</style></head><body>
  <div class="container">
{charts_html}
    <div class="box"><h2>Sumário</h2>{table_html}<div><small>Clique nos cabeçalhos para ordenar ↑↓</small></div></div>
  </div>
</body></html>
"""

    try:
        _cats = ["CAR", "CAR S", "CAR R"]
        labels = "".join(
            f"<label style='margin-right:12px'><input type='checkbox' class='cat-check' data-cat='{c}' checked> {c}</label>"
            for c in _cats
        )
        filters_html = f'<div class="box"><strong>Filtros de Categoria:</strong> {labels}</div>'
        html = html.replace('<div class="container">', '<div class="container">' + filters_html, 1)

        import json as _json

        driver_categories = (
            df.dropna(subset=["Driver"])
              .drop_duplicates(subset=["Driver"])
              .set_index("Driver")["Category"]
              .to_dict()
        )
        _cat_map = {str(k): v for k, v in driver_categories.items()}
        FILTER_JS = f"""<script>
const driverCategory = {_json.dumps(_cat_map, ensure_ascii=False)};
function applyCatFilter() {{
  const selected = {{}};
  document.querySelectorAll('.cat-check').forEach(ch => {{ selected[ch.dataset.cat] = ch.checked; }});
  document.querySelectorAll('.js-plotly-plot').forEach(gd => {{
    if (!gd.data) return;
    const vis = gd.data.map(tr => {{
      const cat = driverCategory[tr.name] || 'OUTROS';
      return (selected[cat] !== false);
    }});
    Plotly.restyle(gd, {{visible: vis}});
  }});
  const rows = document.querySelectorAll('#summary-table tbody tr');
  rows.forEach(tr => {{
    const piloto = (tr.cells[0]||{{}}).textContent?.trim() || '';
    const cat = driverCategory[piloto] || 'OUTROS';
    tr.style.display = (selected[cat] !== false) ? '' : 'none';
  }});
}}
document.addEventListener('change', (e)=>{{ if (e.target && e.target.classList.contains('cat-check')) applyCatFilter(); }});
window.addEventListener('load', applyCatFilter);
</script>
"""
        html = html + SORT_JS + FILTER_JS
    except Exception:
        html = html + SORT_JS

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


def export_report_pdf(lap_df: pd.DataFrame, out_pdf: str = "report.pdf",
                      max_lap_sec=200.0, max_sector_sec=100.0,
                      invert_y=False, hide_out_in: bool = False):
    """Gera um PDF estático com os mesmos gráficos do relatório interativo."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as e:
        raise RuntimeError("Plotly não está instalado. Rode: pip install plotly") from e
    try:
        from PIL import Image
        from io import BytesIO
    except Exception as e:
        raise RuntimeError("Pillow não está instalado. Rode: pip install pillow") from e

    df, order, color_map = _prepare_plotly_dataset(
        lap_df,
        max_lap_sec=max_lap_sec,
        max_sector_sec=max_sector_sec,
        hide_out_in=hide_out_in,
    )
    summary_df = build_summary_df(df, order)

    fig_sections = _build_plotly_sections(go, df, order, color_map, invert_y)
    fig_sections.append(("Sumário", _build_summary_table_figure(go, summary_df, color_map)))

    images = []
    for _, fig in fig_sections:
        try:
            png_bytes = pio.to_image(fig, format="png", width=1600, height=900, scale=2)
        except Exception as e:
            raise RuntimeError("Falha ao exportar imagem com Kaleido. Rode: pip install -U kaleido") from e
        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        images.append(img)

    if not images:
        raise RuntimeError("Nenhuma figura para exportar.")

    first, *rest = images
    first.save(out_pdf, format="PDF", save_all=True, append_images=rest)
    return out_pdf

# ---------- CLI ----------
def _pick_csv_path(initial_dir: str | None = None) -> str:
    """Abre seletor de arquivo para escolher o CSV; cai para input no console se GUI indisponivel."""
    try:
        import tkinter as _tk
        from tkinter import filedialog as _fd
        root = _tk.Tk()
        root.withdraw()
        if initial_dir is None:
            initial_dir = str(Path.cwd())
        path = _fd.askopenfilename(
            title="Selecione o arquivo CSV de timing",
            initialdir=initial_dir,
            filetypes=[("Arquivos CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        try:
            root.update()
        except Exception:
            pass
        root.destroy()
        if path:
            return path
    except Exception:
        pass

    # Fallback console
    try:
        path = input("Informe o caminho do arquivo CSV: ").strip().strip('"')
    except EOFError:
        path = ""
    if not path:
        raise SystemExit("Nenhum arquivo selecionado.")
    return path

def _pick_save_path(title: str,
                    initial_dir: str | None,
                    default_name: str,
                    defaultextension: str,
                    filetypes: list[tuple[str, str]]) -> str:
    """Abre um diálogo 'Salvar como' e retorna o caminho escolhido.
    Se a GUI não estiver disponível, cai para input no console com um valor padrão.
    """
    try:
        import tkinter as _tk
        from tkinter import filedialog as _fd
        root = _tk.Tk()
        root.withdraw()
        if initial_dir is None:
            initial_dir = str(Path.cwd())
        path = _fd.asksaveasfilename(
            title=title,
            initialdir=initial_dir,
            initialfile=default_name,
            defaultextension=defaultextension,
            filetypes=filetypes,
            confirmoverwrite=True,
        )
        try:
            root.update()
        except Exception:
            pass
        root.destroy()
        if path:
            return path
    except Exception:
        pass

    # Fallback console
    default_path = str(Path(initial_dir or ".").resolve() / default_name)
    try:
        text = input(f"Salvar como [{default_path}]: ").strip().strip('"')
    except EOFError:
        text = ""
    return text or default_path

def main():
    ap = argparse.ArgumentParser(description="Gerador de relatório de timing (voltas e setores)")
    ap.add_argument("--csv", required=False, help="Caminho do CSV de timing (se ausente, abre seletor)")
    ap.add_argument("--outdir", default="timing_report", help="Pasta de saída do relatório")
    ap.add_argument("--max-lap-sec", type=float, default=200.0, help="Filtro visual: laps acima disso são ocultados")
    ap.add_argument("--max-sector-sec", type=float, default=100.0, help="Filtro visual de setores")
    ap.add_argument("--pit-gap-sec", type=float, default=25.0, help="Gap (s) acima da mediana para detectar pit (in/out-lap)")
    ap.add_argument("--hide-out-in", action="store_true", help="Oculta out/in-laps nos gráficos")
    ap.add_argument("--invert-y", action="store_true", help="Inverte eixo Y (mais rápido em cima)")
    ap.add_argument("--interactive", action="store_true",
                    help="Gera relatório HTML interativo (Plotly) com hover de coordenadas")
    ap.add_argument("--pdf", action="store_true", help="Exporta PDF estático offline (além do HTML)")
    args = ap.parse_args()

    # Seleciona CSV via CLI ou seletor de arquivo
    csv_path = args.csv or _pick_csv_path()
    if not Path(csv_path).exists():
        raise SystemExit(f"Arquivo CSV nao encontrado: {csv_path}")
    # HTML interativo é sempre gerado; PDF é opcional

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = parse_lap_data(csv_path)
    # Anota stints e tipos de volta (in/out/normal)
    df = annotate_stints(df, pit_gap_sec=args.pit_gap_sec)
    metrics_raw, metrics_show = compute_driver_metrics(df)

    # Pergunta onde salvar o HTML
    html_target = _pick_save_path(
        title="Salvar relatório HTML",
        initial_dir=str(outdir),
        default_name="index.html",
        defaultextension=".html",
        filetypes=[("HTML", "*.html")],
    )
    Path(html_target).parent.mkdir(parents=True, exist_ok=True)

    html_path = generate_report_interactive(df,
                                               out_html=str(html_target),
                                               max_lap_sec=args.max_lap_sec,
                                               max_sector_sec=args.max_sector_sec,
                                               invert_y=args.invert_y,
                                               hide_out_in=args.hide_out_in)
    

    print(f"Relatório gerado em: {html_path}")
    if args.pdf:
        pdf_target = _pick_save_path(
            title="Salvar relatório PDF",
            initial_dir=str(Path(html_target).parent),
            default_name="index.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
        )
        Path(pdf_target).parent.mkdir(parents=True, exist_ok=True)
        pdf_path = export_report_pdf(
            df,
            out_pdf=str(pdf_target),
            max_lap_sec=args.max_lap_sec,
            max_sector_sec=args.max_sector_sec,
            invert_y=args.invert_y,
            hide_out_in=args.hide_out_in,
        )
        print(f"Relatório PDF gerado em: {pdf_path}")

if __name__ == "__main__":
    main()
