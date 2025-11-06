"""
Sistema de Tracking de Vehiculos con Video Real
Uso: python video_real.py --video tu_video.mp4
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import argparse

# Instalar: pip install ultralytics
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


def main():
    parser = argparse.ArgumentParser(description='Sistema de Tracking de Vehiculos con Video Real')
    parser.add_argument('--video', type=str, required=True, help='Ruta al video o 0 para camara')
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
    tracker = Tracker(max_distancia=80, max_frames_perdidos=15)
    mapa = MapaTrayectorias(ancho, alto, decay_rate=0.98)

    frame_count = 0

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

        # Visualizar
        mapa_visual = mapa.obtener_mapa_visual()
        frame = cv2.addWeighted(frame, 0.6, mapa_visual, 0.4, 0)

        # Dibujar trayectorias y bounding boxes
        for track_id, trayectoria in mapa.trayectorias.items():
            if len(trayectoria) > 1:
                if track_id not in colores_track:
                    colores_track[track_id] = colores[len(colores_track) % len(colores)]
                color = colores_track[track_id]
                puntos = np.array(list(trayectoria), dtype=np.int32)
                cv2.polylines(frame, [puntos], False, color, 2)

        # Dibujar IDs de vehiculos
        for track_id, track_data in tracker.tracks.items():
            bbox = track_data['bbox']
            x, y, w, h = bbox
            if track_id in colores_track:
                color = colores_track[track_id]
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (int(x), int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Info
        info = f"Frame: {frame_count} | Tracks: {len(tracker.tracks)}"
        cv2.putText(frame, info, (10, alto - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Tracking de Vehiculos - Video Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Completado")


if __name__ == "__main__":
    main()
