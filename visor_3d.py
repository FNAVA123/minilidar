"""Generate a fully local, interactive 3D point-cloud and aerial viewer."""
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def create_viewer(output, video_name="Video"):
    output = Path(output)
    summary = json.loads((output / "resumen.json").read_text())
    cloud = o3d.io.read_point_cloud(str(output / "nube_3d.ply"))
    xyz, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    if not len(xyz):
        raise ValueError("La nube no contiene puntos para visualizar.")
    ids = np.random.default_rng(0).choice(len(xyz), min(len(xyz), 180000), replace=False)
    points = xyz[ids]
    if len(colors) == len(xyz):
        rgb = np.clip(colors[ids] * 255, 0, 255).astype(np.uint8)
        color = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    else:
        color = "#4974c8"
    with (output / "trayectoria.csv").open() as file:
        rows = list(csv.DictReader(file))
    path = np.array([[float(row[k]) for k in ("x", "y", "z")] for row in rows])
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 2], z=-points[:, 1], mode="markers",
                              marker=dict(size=2, color=color, opacity=1), hoverinfo="skip",
                              name="Nube 3D", showlegend=False))
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 2], z=-path[:, 1], mode="lines",
                              line=dict(color="#e84a5f", width=5), name="Recorrido"))
    fig.add_trace(go.Scatter3d(x=[path[0, 0]], y=[path[0, 2]], z=[-path[0, 1]], mode="markers",
                              marker=dict(color="#21b573", size=6), name="Inicio"))
    fig.update_layout(template="plotly_white", autosize=True,
                      margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(aspectmode="data", dragmode="orbit",
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=.9)),
                                 xaxis_title="X (m estimados)", yaxis_title="Z (m estimados)",
                                 zaxis_title="Altura (m estimados)"),
                      legend=dict(orientation="h", x=0, y=1), paper_bgcolor="white")
    chart = fig.to_html(full_html=False, include_plotlyjs=True,
                        config=dict(displaylogo=False, scrollZoom=True, responsive=True),
                        default_height="100%", default_width="100%")
    aerial = base64.b64encode((output / "vista_superior.png").read_bytes()).decode("ascii")
    reason = summary["stop_reason"]
    status = "Video procesado" if reason == "end_of_video" else "Reconstrucción parcial"
    explanations = {"tracking_lost": "Se guardó lo reconstruido antes de perder el seguimiento.",
                    "point_budget": "Se alcanzó el límite de puntos configurado.",
                    "frame_limit": "La ejecución se limitó a una prueba corta.",
                    "end_of_video": "Se llegó al final del video."}
    count = f"{summary['points_saved']:,}"
    text = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniLiDAR · {html.escape(video_name)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f9;color:#172236;font:15px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
header{{padding:22px 28px;background:white;border-bottom:1px solid #dfe5ef}}h1{{font-size:23px;margin:0 0 8px}}header p{{margin:5px 0;color:#526078}}
.badge{{display:inline-block;padding:4px 10px;border-radius:20px;background:#eaf0ff;color:#284b9b;font-size:13px}}
main{{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(320px,1fr);gap:16px;padding:18px}}
section{{background:white;border:1px solid #dfe5ef;border-radius:12px;overflow:hidden;min-width:0}}h2{{font-size:17px;margin:0;padding:16px 18px 4px}}
.hint{{margin:5px 18px 12px;color:#627088;font-size:13px}}.chart{{height:65vh;min-height:380px}}.chart>div{{height:100%}}.aerial{{width:100%;height:58vh;object-fit:contain}}
footer{{padding:0 28px 22px;color:#627088;font-size:13px}}a{{color:#284b9b;margin-right:18px}}@media(max-width:900px){{main{{grid-template-columns:1fr}}.chart{{height:60vh}}.aerial{{height:auto}}}}
body{{height:100dvh;display:flex;flex-direction:column;overflow:hidden}}
header{{flex:none;padding:10px 18px}}h1{{font-size:19px}}header p{{display:inline;margin-right:14px;font-size:12px}}
main{{flex:1;min-height:0;grid-template-columns:minmax(0,63fr) minmax(0,37fr);gap:8px;padding:8px}}
section{{display:flex;flex-direction:column;min-height:0}}h2{{padding:10px 12px 4px}}
.hint{{margin:4px 12px 8px}}.chart{{flex:1;min-height:0;height:auto}}.aerial{{flex:1;min-height:0;height:0;object-fit:contain}}
footer{{flex:none;padding:0 16px 5px;font-size:11px}}footer p{{margin:4px 0}}
button{{float:right;border:1px solid #cdd8e9;border-radius:8px;background:#eaf0ff;color:#284b9b;padding:8px 12px;cursor:pointer}}
@media(max-width:700px){{body{{height:auto;min-height:100dvh;overflow:auto}}main{{grid-template-columns:1fr;flex:none}}.chart{{height:60vh;flex:none}}.aerial{{height:45vh;flex:none}}}}
</style></head><body>
<header><button id="fullscreen" type="button">Pantalla completa</button><h1>MiniLiDAR · {html.escape(video_name)}</h1><span class="badge">{status}</span>
<p>{count} puntos · {summary['accepted_frames']:,} fotogramas aceptados</p>
<p>{explanations.get(reason, 'Los resultados se guardaron en esta carpeta.')}</p></header>
<main><section><h2>Mapa 3D</h2><p class="hint">Arrastra para girar · Rueda para acercar · Mayús + arrastrar para desplazar</p>
<div class="chart">{chart}</div></section><section><h2>Vista aérea</h2>
<p class="hint">Proyección del entorno y recorrido estimado de la cámara</p>
<img class="aerial" alt="Vista aérea reconstruida" src="data:image/png;base64,{aerial}"></section></main>
<footer><p><a href="nube_3d.ply" download>Guardar nube 3D (.ply)</a><a href="trayectoria.csv" download>Guardar trayectoria</a><a href="vista_superior.png">Abrir vista aérea</a></p>
<p>Escala aproximada. La vista aérea no es un mapa de ocupación ni está alineada automáticamente al suelo.
La vista 3D muestra una muestra de hasta 180,000 puntos; el archivo PLY conserva todos los puntos exportados.</p>
</footer><script>
const resizeCharts=()=>document.querySelectorAll('.js-plotly-plot').forEach(el=>Plotly.Plots.resize(el));
new ResizeObserver(resizeCharts).observe(document.querySelector('main'));
document.getElementById('fullscreen').onclick=async()=>{{
 try {{if(document.fullscreenElement) await document.exitFullscreen();else await document.documentElement.requestFullscreen();}}
 catch(e) {{document.getElementById('fullscreen').textContent='Usa Control + Cmd + F';}}
}};
document.addEventListener('fullscreenchange',()=>{{document.getElementById('fullscreen').textContent=document.fullscreenElement?'Salir de pantalla completa':'Pantalla completa';resizeCharts();}});
</script></body></html>"""
    viewer = output / "mapa_3d.html"
    viewer.write_text(text, encoding="utf-8")
    return viewer
