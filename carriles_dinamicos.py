"""
Sistema de Detección Dinámica de Carriles mediante Tendencias de Trayectorias
Uso: python carriles_dinamicos.py --video tu_video.mp4
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import argparse
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist

# Instalar: pip install ultralytics scipy
from ultralytics import YOLO


class Tracker:
    """Tracker mejorado con IDs persistentes"""
    def __init__(self, max_distancia=80, max_frames_perdidos=15):
        self.siguiente_id = 0
        self.tracks = {}
        self.max_distancia = max_distancia
        self.max_frames_perdidos = max_frames_perdidos

    def actualizar(self, detecciones):
        if not detecciones:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['frames_perdidos'] += 1
                if self.tracks[track_id]['frames_perdidos'] > self.max_frames_perdidos:
                    del self.tracks[track_id]
            return []

        tracks_actualizados = []
        detecciones_usadas = set()

        for track_id, track_data in list(self.tracks.items()):
            mejor_dist = float('inf')
            mejor_idx = -1

            for i, det in enumerate(detecciones):
                if i in detecciones_usadas:
                    continue
                x, y, w, h = det
                centro = (x + w/2, y + h/2)

                dist = np.sqrt((track_data['centro'][0] - centro[0])**2 +
                             (track_data['centro'][1] - centro[1])**2)

                if dist < mejor_dist and dist < self.max_distancia:
                    mejor_dist = dist
                    mejor_idx = i

            if mejor_idx != -1:
                x, y, w, h = detecciones[mejor_idx]
                centro = (x + w/2, y + h/2)
                self.tracks[track_id] = {
                    'bbox': detecciones[mejor_idx],
                    'centro': centro,
                    'frames_perdidos': 0
                }
                detecciones_usadas.add(mejor_idx)
                tracks_actualizados.append((track_id, x, y, w, h))
            else:
                self.tracks[track_id]['frames_perdidos'] += 1

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['frames_perdidos'] > self.max_frames_perdidos:
                del self.tracks[track_id]

        for i, det in enumerate(detecciones):
            if i not in detecciones_usadas:
                track_id = self.siguiente_id
                self.siguiente_id += 1
                x, y, w, h = det
                centro = (x + w/2, y + h/2)
                self.tracks[track_id] = {
                    'bbox': det,
                    'centro': centro,
                    'frames_perdidos': 0
                }
                tracks_actualizados.append((track_id, x, y, w, h))

        return tracks_actualizados


class MapaTrayectorias:
    def __init__(self, ancho, alto, decay_rate=0.98):
        self.ancho = ancho
        self.alto = alto
        self.mapa = np.zeros((alto, ancho), dtype=np.float32)
        self.decay_rate = decay_rate
        self.trayectorias = defaultdict(lambda: deque(maxlen=100))

    def actualizar(self, tracks):
        self.mapa *= self.decay_rate

        for track_id, x, y, w, h in tracks:
            centro_x = int(x + w/2)
            centro_y = int(y + h/2)

            if len(self.trayectorias[track_id]) > 0:
                ultima_pos = self.trayectorias[track_id][-1]
                dist = np.sqrt((centro_x - ultima_pos[0])**2 + (centro_y - ultima_pos[1])**2)
                if dist > 5:
                    self.trayectorias[track_id].append((centro_x, centro_y))
            else:
                self.trayectorias[track_id].append((centro_x, centro_y))

            trayectoria = list(self.trayectorias[track_id])
            if len(trayectoria) > 1:
                for i in range(len(trayectoria) - 1):
                    pt1 = trayectoria[i]
                    pt2 = trayectoria[i + 1]
                    cv2.line(self.mapa, pt1, pt2, 1.0, 3)

    def obtener_mapa_visual(self):
        mapa_norm = cv2.normalize(self.mapa, None, 0, 255, cv2.NORM_MINMAX)
        mapa_norm = mapa_norm.astype(np.uint8)
        mapa_color = cv2.applyColorMap(mapa_norm, cv2.COLORMAP_HOT)
        return mapa_color


class AnalizadorCarrilesDinamicos:
    """Analiza trayectorias para detectar carriles dinámicamente mediante líneas de tendencia"""
    def __init__(self, min_puntos_trayectoria=20, distancia_agrupacion=50):
        self.trayectorias_completas = []  # Trayectorias completadas/largas
        self.min_puntos_trayectoria = min_puntos_trayectoria
        self.distancia_agrupacion = distancia_agrupacion
        self.carriles = []  # Lista de carriles con sus líneas de tendencia
        self.ultimo_calculo = 0

    def agregar_trayectoria(self, trayectoria):
        """Agrega una trayectoria completa si tiene suficientes puntos"""
        if len(trayectoria) >= self.min_puntos_trayectoria:
            trayectoria_array = np.array(list(trayectoria))
            self.trayectorias_completas.append(trayectoria_array)
            # Mantener solo las últimas 50 trayectorias para no saturar
            if len(self.trayectorias_completas) > 50:
                self.trayectorias_completas.pop(0)

    def calcular_similitud_trayectorias(self, traj1, traj2):
        """Calcula la distancia promedio entre dos trayectorias"""
        if len(traj1) < 5 or len(traj2) < 5:
            return float('inf')

        # Muestrear puntos para comparación
        indices1 = np.linspace(0, len(traj1) - 1, min(10, len(traj1))).astype(int)
        indices2 = np.linspace(0, len(traj2) - 1, min(10, len(traj2))).astype(int)

        puntos1 = traj1[indices1]
        puntos2 = traj2[indices2]

        # Calcular distancia promedio entre trayectorias
        distancias = cdist(puntos1, puntos2)
        return np.min(distancias, axis=1).mean()

    def agrupar_trayectorias(self):
        """Agrupa trayectorias similares (mismo carril)"""
        if len(self.trayectorias_completas) < 2:
            return []

        grupos = []
        trayectorias_usadas = set()

        for i, traj1 in enumerate(self.trayectorias_completas):
            if i in trayectorias_usadas:
                continue

            grupo_actual = [traj1]
            trayectorias_usadas.add(i)

            for j, traj2 in enumerate(self.trayectorias_completas):
                if j <= i or j in trayectorias_usadas:
                    continue

                similitud = self.calcular_similitud_trayectorias(traj1, traj2)

                if similitud < self.distancia_agrupacion:
                    grupo_actual.append(traj2)
                    trayectorias_usadas.add(j)

            if len(grupo_actual) >= 1:  # Al menos 1 trayectoria por grupo
                grupos.append(grupo_actual)

        return grupos

    def calcular_linea_tendencia(self, trayectorias_grupo):
        """Calcula una línea de tendencia suave para un grupo de trayectorias"""
        # Combinar todos los puntos del grupo
        todos_puntos = []
        for traj in trayectorias_grupo:
            todos_puntos.extend(traj)

        todos_puntos = np.array(todos_puntos)

        if len(todos_puntos) < 10:
            return None

        # Ordenar por coordenada Y (vertical)
        todos_puntos = todos_puntos[todos_puntos[:, 1].argsort()]

        # Dividir en segmentos por Y y promediar X
        y_min, y_max = todos_puntos[:, 1].min(), todos_puntos[:, 1].max()
        num_segmentos = 15
        y_bins = np.linspace(y_min, y_max, num_segmentos)

        puntos_promedio = []
        for i in range(len(y_bins) - 1):
            mask = (todos_puntos[:, 1] >= y_bins[i]) & (todos_puntos[:, 1] < y_bins[i + 1])
            puntos_bin = todos_puntos[mask]

            if len(puntos_bin) > 0:
                x_promedio = np.median(puntos_bin[:, 0])
                y_promedio = (y_bins[i] + y_bins[i + 1]) / 2
                puntos_promedio.append([x_promedio, y_promedio])

        if len(puntos_promedio) < 4:
            return None

        puntos_promedio = np.array(puntos_promedio)

        # Crear spline suave
        try:
            # Ordenar por Y
            puntos_promedio = puntos_promedio[puntos_promedio[:, 1].argsort()]

            # Crear spline
            k = min(3, len(puntos_promedio) - 1)
            spline = UnivariateSpline(puntos_promedio[:, 1], puntos_promedio[:, 0], k=k, s=1000)

            # Generar puntos suaves
            y_vals = np.linspace(puntos_promedio[:, 1].min(), puntos_promedio[:, 1].max(), 50)
            x_vals = spline(y_vals)

            linea_tendencia = np.column_stack([x_vals, y_vals])
            return linea_tendencia
        except:
            return puntos_promedio

    def actualizar_carriles(self, frame_count):
        """Actualiza el cálculo de carriles cada cierto tiempo"""
        # Recalcular cada 30 frames
        if frame_count - self.ultimo_calculo < 30:
            return False

        self.ultimo_calculo = frame_count

        # Agrupar trayectorias
        grupos = self.agrupar_trayectorias()

        # Calcular líneas de tendencia para cada grupo
        self.carriles = []
        for grupo in grupos:
            linea = self.calcular_linea_tendencia(grupo)
            if linea is not None:
                # Calcular posición X promedio para ordenar carriles
                x_promedio = np.mean(linea[:, 0])
                self.carriles.append({
                    'linea': linea,
                    'x_promedio': x_promedio,
                    'num_trayectorias': len(grupo)
                })

        # Ordenar carriles de izquierda a derecha
        self.carriles.sort(key=lambda c: c['x_promedio'])

        return len(self.carriles) > 0

    def asignar_carril(self, x, y):
        """Asigna un vehículo al carril más cercano"""
        if not self.carriles:
            return None

        mejor_carril = None
        mejor_distancia = float('inf')

        for i, carril in enumerate(self.carriles):
            linea = carril['linea']

            # Encontrar el punto más cercano en la línea
            distancias = np.sqrt(np.sum((linea - np.array([x, y]))**2, axis=1))
            distancia_min = np.min(distancias)

            if distancia_min < mejor_distancia:
                mejor_distancia = distancia_min
                mejor_carril = i + 1

        # Solo asignar si está lo suficientemente cerca (menos de 80 píxeles)
        if mejor_distancia < 80:
            return mejor_carril
        return None


def main():
    parser = argparse.ArgumentParser(description='Sistema de Detección Dinámica de Carriles')
    parser.add_argument('--video', type=str, required=True, help='Ruta al video o 0 para camara')
    parser.add_argument('--modelo', type=str, default='yolov8n.pt', help='Modelo YOLO a usar')
    parser.add_argument('--min-puntos', type=int, default=20, help='Mínimo de puntos para trayectoria válida')
    args = parser.parse_args()

    # Cargar YOLO
    print("Cargando YOLO...")
    modelo = YOLO(args.modelo)
    print("YOLO cargado")

    # Abrir video
    video_path = args.video if args.video != '0' else 0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir {args.video}")
        return

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {ancho}x{alto}")
    print("Procesando...")

    # Componentes
    tracker = Tracker(max_distancia=80, max_frames_perdidos=15)
    mapa = MapaTrayectorias(ancho, alto, decay_rate=0.98)
    analizador = AnalizadorCarrilesDinamicos(min_puntos_trayectoria=args.min_puntos)

    frame_count = 0
    carriles_detectados = False
    tracks_completados = {}  # Para guardar trayectorias completas

    colores = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
               (255, 255, 100), (255, 100, 255), (100, 255, 255)]
    colores_track = {}
    colores_carril = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detectar con YOLO
        resultados = modelo(frame, verbose=False, conf=0.5)
        detecciones = []

        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    detecciones.append([x, y, w, h])

        # Tracking
        tracks_anteriores = set(tracker.tracks.keys())
        tracks = tracker.actualizar(detecciones)
        tracks_actuales = set(tracker.tracks.keys())

        # Detectar tracks que terminaron
        tracks_finalizados = tracks_anteriores - tracks_actuales
        for track_id in tracks_finalizados:
            if track_id in mapa.trayectorias:
                analizador.agregar_trayectoria(mapa.trayectorias[track_id])

        # Actualizar mapa
        mapa.actualizar(tracks)

        # Actualizar carriles dinámicamente
        if analizador.actualizar_carriles(frame_count):
            carriles_detectados = True

        # Verificar si tenemos suficientes carriles detectados (al menos 2)
        carriles_activos = carriles_detectados and len(analizador.carriles) >= 2

        # Visualizar mapa de calor solo si no hay carriles detectados
        if not carriles_activos:
            # Modo de aprendizaje: mostrar mapa de calor con trayectorias
            mapa_visual = mapa.obtener_mapa_visual()
            frame = cv2.addWeighted(frame, 0.7, mapa_visual, 0.3, 0)
        else:
            # Modo de carriles: mapa de calor muy tenue o sin él
            mapa_visual = mapa.obtener_mapa_visual()
            frame = cv2.addWeighted(frame, 0.95, mapa_visual, 0.05, 0)

        # Dibujar líneas de tendencia de carriles (cuando estén disponibles)
        if carriles_activos:
            for i, carril in enumerate(analizador.carriles):
                linea = carril['linea'].astype(np.int32)
                color = colores_carril[i % len(colores_carril)]

                # Dibujar línea gruesa y prominente
                cv2.polylines(frame, [linea], False, color, 6)

                # Etiqueta del carril
                if len(linea) > 0:
                    punto_medio = linea[len(linea) // 2]
                    cv2.putText(frame, f"Carril {i+1}",
                               (punto_medio[0] - 40, punto_medio[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Dibujar trayectorias activas SOLO si no hay carriles detectados
        if not carriles_activos:
            for track_id, trayectoria in mapa.trayectorias.items():
                if len(trayectoria) > 1:
                    if track_id not in colores_track:
                        colores_track[track_id] = colores[len(colores_track) % len(colores)]
                    color = colores_track[track_id]
                    puntos = np.array(list(trayectoria), dtype=np.int32)
                    cv2.polylines(frame, [puntos], False, color, 2)

        # Dibujar vehículos con asignación de carril
        for track_id, track_data in tracker.tracks.items():
            bbox = track_data['bbox']
            x, y, w, h = bbox
            centro_x = int(x + w/2)
            centro_y = int(y + h/2)

            if track_id in colores_track:
                color = colores_track[track_id]
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)

                # Asignar carril
                carril_num = analizador.asignar_carril(centro_x, centro_y)
                if carril_num:
                    cv2.putText(frame, f"ID:{track_id} C{carril_num}", (int(x), int(y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, f"ID:{track_id}", (int(x), int(y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Indicador de modo
        if carriles_activos:
            modo_texto = "MODO: CARRILES ACTIVOS"
            modo_color = (0, 255, 0)  # Verde
        else:
            modo_texto = "MODO: APRENDIZAJE"
            modo_color = (0, 255, 255)  # Amarillo

        cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 50), modo_color, 2)
        cv2.putText(frame, modo_texto, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, modo_color, 2)

        # Info
        info = f"Frame: {frame_count} | Tracks: {len(tracker.tracks)} | Carriles: {len(analizador.carriles)} | Trayectorias: {len(analizador.trayectorias_completas)}"
        cv2.putText(frame, info, (10, alto - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Carriles Dinamicos - Lineas de Tendencia', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Completado")


if __name__ == "__main__":
    main()
