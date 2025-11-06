"""
Sistema de Corredores con Video Real
Uso: python video_real.py --video tu_video.mp4
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import KMeans
import argparse

# Instalar: pip install ultralytics
from ultralytics import YOLO


class TrackerSimple:
    def __init__(self, max_distancia=100, max_frames_perdidos=30):
        self.siguiente_id = 0
        self.tracks = {}
        self.max_distancia = max_distancia
        self.max_frames_perdidos = max_frames_perdidos

    def actualizar(self, detecciones):
        centros_detecciones = []
        for det in detecciones:
            x, y, w, h = det
            centro = (x + w/2, y + h/2)
            centros_detecciones.append(centro)

        if not self.tracks:
            tracks_actualizados = []
            for i, det in enumerate(detecciones):
                track_id = self.siguiente_id
                self.siguiente_id += 1
                x, y, w, h = det
                self.tracks[track_id] = {
                    'bbox': det,
                    'centro': centros_detecciones[i],
                    'frames_perdidos': 0
                }
                tracks_actualizados.append((track_id, x, y, w, h))
            return tracks_actualizados

        tracks_actualizados = []
        detecciones_asignadas = set()

        for track_id, track_data in list(self.tracks.items()):
            mejor_dist = float('inf')
            mejor_idx = -1

            for i, centro_det in enumerate(centros_detecciones):
                if i in detecciones_asignadas:
                    continue
                dist = np.sqrt((track_data['centro'][0] - centro_det[0])**2 +
                             (track_data['centro'][1] - centro_det[1])**2)
                if dist < mejor_dist and dist < self.max_distancia:
                    mejor_dist = dist
                    mejor_idx = i

            if mejor_idx != -1:
                det = detecciones[mejor_idx]
                self.tracks[track_id] = {
                    'bbox': det,
                    'centro': centros_detecciones[mejor_idx],
                    'frames_perdidos': 0
                }
                detecciones_asignadas.add(mejor_idx)
                x, y, w, h = det
                tracks_actualizados.append((track_id, x, y, w, h))
            else:
                self.tracks[track_id]['frames_perdidos'] += 1

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['frames_perdidos'] > self.max_frames_perdidos:
                del self.tracks[track_id]

        for i, det in enumerate(detecciones):
            if i not in detecciones_asignadas:
                track_id = self.siguiente_id
                self.siguiente_id += 1
                self.tracks[track_id] = {
                    'bbox': det,
                    'centro': centros_detecciones[i],
                    'frames_perdidos': 0
                }
                x, y, w, h = det
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


class AnalizadorCorredores:
    def __init__(self, num_corredores=3):
        self.num_corredores = num_corredores
        self.puntos_trayectoria = deque(maxlen=1000)
        self.kmeans = None
        self.centros = None
        self.min_puntos = 100

    def agregar_trayectorias(self, mapa_trayectorias):
        for track_id, trayectoria in mapa_trayectorias.trayectorias.items():
            for punto in trayectoria:
                self.puntos_trayectoria.append(punto)

    def calcular_corredores(self):
        if len(self.puntos_trayectoria) < self.min_puntos:
            return False

        puntos = np.array(list(self.puntos_trayectoria))
        self.kmeans = KMeans(n_clusters=self.num_corredores, random_state=42, n_init=10)
        self.kmeans.fit(puntos)
        self.centros = self.kmeans.cluster_centers_
        self.centros = self.centros[self.centros[:, 0].argsort()]
        return True

    def asignar_corredor(self, x, y):
        if self.kmeans is None:
            return None
        cluster = self.kmeans.predict([[x, y]])[0]
        for i, centro in enumerate(self.centros):
            if np.array_equal(centro, self.kmeans.cluster_centers_[cluster]):
                return i + 1
        return cluster + 1

    def obtener_limites(self):
        if self.centros is None or len(self.centros) < 2:
            return []
        limites = []
        centros_x = sorted(self.centros[:, 0])
        for i in range(len(centros_x) - 1):
            limite = int((centros_x[i] + centros_x[i + 1]) / 2)
            limites.append(limite)
        return limites


def main():
    parser = argparse.ArgumentParser(description='Sistema de Corredores con Video Real')
    parser.add_argument('--video', type=str, required=True, help='Ruta al video o 0 para camara')
    parser.add_argument('--corredores', type=int, default=3, help='Numero de corredores a detectar')
    parser.add_argument('--modelo', type=str, default='yolov8n.pt', help='Modelo YOLO a usar')
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
    tracker = TrackerSimple(max_distancia=100, max_frames_perdidos=30)
    mapa = MapaTrayectorias(ancho, alto, decay_rate=0.98)
    analizador = AnalizadorCorredores(args.corredores)

    frame_count = 0
    corredores_detectados = False

    colores = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
               (255, 255, 100), (255, 100, 255), (100, 255, 255)]
    colores_track = {}

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
        tracks = tracker.actualizar(detecciones)

        # Actualizar mapa
        mapa.actualizar(tracks)
        analizador.agregar_trayectorias(mapa)

        # Calcular corredores cada 20 frames
        if frame_count % 20 == 0:
            if analizador.calcular_corredores():
                corredores_detectados = True

        # Visualizar
        mapa_visual = mapa.obtener_mapa_visual()
        frame = cv2.addWeighted(frame, 0.6, mapa_visual, 0.4, 0)

        # Dibujar trayectorias
        for track_id, trayectoria in mapa.trayectorias.items():
            if len(trayectoria) > 1:
                if track_id not in colores_track:
                    colores_track[track_id] = colores[len(colores_track) % len(colores)]
                color = colores_track[track_id]
                puntos = np.array(list(trayectoria), dtype=np.int32)
                cv2.polylines(frame, [puntos], False, color, 2)

        # Dibujar corredores
        if corredores_detectados and analizador.centros is not None:
            limites = analizador.obtener_limites()
            for limite_x in limites:
                cv2.line(frame, (limite_x, 0), (limite_x, alto), (255, 0, 255), 2)

            for i, centro in enumerate(analizador.centros):
                x, y = int(centro[0]), int(centro[1])
                cv2.circle(frame, (x, y), 12, (0, 255, 255), -1)
                cv2.putText(frame, f"C{i+1}", (x-12, y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"Corredor {i+1}", (x-50, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Asignar vehiculos
            for track_id, track_data in tracker.tracks.items():
                bbox = track_data['bbox']
                x, y, w, h = bbox
                centro_x = int(x + w/2)
                centro_y = int(y + h/2)
                corredor = analizador.asignar_corredor(centro_x, centro_y)
                if corredor:
                    cv2.putText(frame, f"C{corredor}", (int(x), int(y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Info
        info = f"Frame: {frame_count} | Tracks: {len(tracker.tracks)} | Puntos: {len(analizador.puntos_trayectoria)}"
        cv2.putText(frame, info, (10, alto - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Corredores Virtuales - Video Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Completado")


if __name__ == "__main__":
    main()
