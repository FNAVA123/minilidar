"""Local launcher: video path -> reconstruction -> offline 3D/aerial viewer."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


def input_path(raw):
    raw = raw.strip()
    candidate = Path(os.path.expandvars(raw)).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = []
    if len(parts) == 1:
        candidate = Path(os.path.expandvars(parts[0])).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"No se encontró el archivo: {raw}")


def command_for(video, output, settings, calibration=None, max_frames=0):
    command = [sys.executable, str(ROOT / "mini_lidar_orb_depth.py"), str(video),
               "--features", "orb", "--scene", settings["scene"], "--output", str(output)]
    if calibration:
        command += ["--calibration", str(calibration)]
    else:
        command += ["--fov-deg", str(settings["fov_deg"])]
    for key in ("width", "stride", "input_size", "voxel", "pixel_step", "max_points", "recovery_seconds"):
        command += ["--" + key.replace("_", "-"), str(settings[key])]
    # Keep the tested indoor and outdoor ranges rather than guessing the scene.
    command += ["--max-depth", "70" if settings["scene"] == "outdoor" else "15",
                "--max-step", "10" if settings["scene"] == "outdoor" else "5"]
    if max_frames:
        command += ["--max-frames", str(max_frames)]
    return command


def open_viewer(path):
    import webbrowser
    webbrowser.open(Path(path).resolve().as_uri())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruye un video y abre el mapa 3D y la vista aérea.")
    parser.add_argument("video", nargs="?", help="Ruta del video; también puedes arrastrarlo a la terminal")
    scene = parser.add_mutually_exclusive_group()
    scene.add_argument("--exterior", action="store_true", help="Usar el modelo de exteriores")
    scene.add_argument("--interior", action="store_true", help="Usar el modelo de interiores (predeterminado)")
    parser.add_argument("--calibracion", help="Archivo YAML de calibración de la cámara, si lo tienes")
    parser.add_argument("--abrir-ultimo", action="store_true", help="Abrir el último resultado sin procesar de nuevo")
    parser.add_argument("--sin-abrir", action="store_true", help="Guardar resultados sin abrir el navegador")
    parser.add_argument("--max-fotogramas", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_fotogramas < 0:
        parser.error("El límite de fotogramas debe ser positivo")
    try:
        if args.abrir_ultimo:
            pointer = ROOT / "ultimo_resultado.txt"
            if not pointer.is_file():
                raise ValueError("Todavía no hay resultados. Ejecuta ./mapear con un video primero.")
            viewer = ROOT / pointer.read_text().strip() / "mapa_3d.html"
            if not viewer.is_file():
                raise ValueError("El último resultado ya no está en su carpeta.")
            from visor_3d import create_viewer
            viewer=create_viewer(viewer.parent,viewer.parent.name)
            open_viewer(viewer)
            return 0
        raw = args.video or input("Pega la ruta del video o arrástralo aquí y pulsa Enter:\n> ")
        video = input_path(raw)
        calibration = input_path(args.calibracion) if args.calibracion else None
        settings = json.loads((ROOT / "configuracion.json").read_text())
        if args.exterior:
            settings["scene"] = "outdoor"
        elif args.interior:
            settings["scene"] = "indoor"
        if settings["scene"] not in ("indoor", "outdoor"):
            raise ValueError('En configuracion.json, "scene" debe ser "indoor" u "outdoor".')
        stamp = datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y%m%d_%H%M%S")
        output = ROOT / "resultados" / f"{video.stem}_{stamp}_{uuid.uuid4().hex[:6]}"
        output.mkdir(parents=True)
        command = command_for(video, output, settings, calibration, args.max_fotogramas)
        (output / "ejecucion.json").write_text(json.dumps({"video": str(video), "settings": settings,
                                                        "command": command}, indent=2) + "\n")
        print(f"Video: {video.name}\nEscena: {'exterior' if settings['scene'] == 'outdoor' else 'interior'}", flush=True)
        if calibration is None:
            print(f"Sin calibración: campo de visión aproximado de {settings['fov_deg']}°.\n", flush=True)
        with (output / "ejecucion.log").open("w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, cwd=ROOT)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                status = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                print("\nProceso detenido.")
                return 130
        if status not in (0, 2) or not (output / "resumen.json").is_file():
            print(f"No se pudo terminar la reconstrucción. Registro: {output / 'ejecucion.log'}", file=sys.stderr)
            return 1
        from visor_3d import create_viewer
        viewer = create_viewer(output, video.name)
        (ROOT / "ultimo_resultado.txt").write_text(str(output.relative_to(ROOT)) + "\n")
        summary = json.loads((output / "resumen.json").read_text())
        partial = summary["stop_reason"] != "end_of_video"
        print(f"\n{'Resultado parcial' if partial else 'Procesamiento terminado'}: {summary['points_saved']:,} puntos.")
        print(f"Resultados: {output}", flush=True)
        if not args.sin_abrir:
            open_viewer(viewer)
        # Tracking loss is a usable partial result, not a launcher failure.
        return 0
    except (ValueError, FileNotFoundError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
