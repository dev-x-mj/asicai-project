"""
Cuánto mide un píxel térmico en el suelo/equipo (GSD) para el DJI Mavic 3T.

Datos tomados del datasheet (cámara térmica): 640 x 512, pixel 12 micras, DFOV 61 grados.
El datasheet NO trae la distancia focal real: el IFOV se DERIVA del DFOV (que viene
redondeado), así que es una aproximación, no un dato oficial.

Uso:  python src/gsd.py
"""
import math

W, H = 640, 512
DFOV_DEG = 61.0

diag_px = math.hypot(W, H)
tan_half = math.tan(math.radians(DFOV_DEG / 2))
IFOV_RAD = 2 * tan_half / diag_px                       # radianes por pixel (aprox.)
HFOV_DEG = 2 * math.degrees(math.atan(tan_half * W / diag_px))
VFOV_DEG = 2 * math.degrees(math.atan(tan_half * H / diag_px))


def gsd_cm(dist_m: float) -> float:
    """Tamaño de un píxel (cm) a dist_m metros del objetivo."""
    return IFOV_RAD * dist_m * 100


def min_object_cm(dist_m: float, px: int = 3) -> float:
    """Tamaño mínimo de un objeto (cm) para que ocupe 'px' píxeles de lado."""
    return gsd_cm(dist_m) * px


def hotspot_min_area_to_cm(dist_m: float, min_area_px: int = 20) -> float:
    """Diámetro (cm) de un punto circular que ocupa exactamente min_area_px."""
    d_px = 2 * math.sqrt(min_area_px / math.pi)
    return gsd_cm(dist_m) * d_px


if __name__ == "__main__":
    print(f"IFOV derivado: {IFOV_RAD*1000:.3f} mrad/px | HFOV~{HFOV_DEG:.1f} deg, VFOV~{VFOV_DEG:.1f} deg")
    print(f"{'dist(m)':>8} {'pixel(cm)':>10} {'campo ancho(m)':>15} {'obj 3px(cm)':>12} {'obj 5px(cm)':>12} {'area20px=diam(cm)':>18}")
    for d in (3, 5, 10, 15, 20, 30, 50):
        ancho = 2 * d * math.tan(math.radians(HFOV_DEG / 2))
        print(f"{d:>8} {gsd_cm(d):>10.2f} {ancho:>15.1f} {min_object_cm(d,3):>12.1f} {min_object_cm(d,5):>12.1f} {hotspot_min_area_to_cm(d):>18.1f}")
