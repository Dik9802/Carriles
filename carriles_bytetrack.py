"""
Sistema de Carriles Dinámicos - ByteTrack (Sin Kalman Filter)
Mejoras:
1. ByteTrack simplificado: Usa detecciones de alta/baja confianza con IoU matching
2. Dos ventanas: Modo Aprendizaje y Carriles Activos
3. ROI interactivo
4. Validación de carriles: elimina cruces y diagonales
5. Fusión de carriles para estabilidad
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import argparse
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist

from ultralytics import YOLO


def calcular_iou(bbox1, bbox2):
    """Calcula Intersection over Union entre dos bboxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convertir a coordenadas [x1, y1, x2, y2]
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]

    # Calcular intersección
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    interseccion = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - interseccion

    if union == 0:
        return 0.0

    return interseccion / union


class ByteTrackSimple:
    """
    ByteTrack SIN Kalman - Solo usa IoU matching
    - Alta confianza (>0.5): matching primario
    - Baja confianza (0.1-0.5): rescata tracks perdidos
    """
    def __init__(self, high_thresh=0.5, low_thresh=0.1, max_frames_perdidos=30):
        self.siguiente_id = 0
        self.tracks = {}
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_frames_perdidos = max_frames_perdidos

    def actualizar(self, detecciones_con_conf):
        """
        detecciones_con_conf: lista de [x, y, w, h, conf]
        """
        # Separar detecciones por confianza
        detecciones_altas = []
        detecciones_bajas = []

        for det in detecciones_con_conf:
            x, y, w, h, conf = det
            if conf >= self.high_thresh:
                detecciones_altas.append([x, y, w, h, conf])
            elif conf >= self.low_thresh:
                detecciones_bajas.append([x, y, w, h, conf])

        # === PRIMER MATCHING: Alta confianza ===
        tracks_asignados = set()
        detecciones_usadas_altas = set()
        tracks_actualizados = []

        for track_id, track_data in list(self.tracks.items()):
            mejor_iou = 0
            mejor_idx = -1
            bbox_anterior = track_data['bbox']

            for i, det in enumerate(detecciones_altas):
                if i in detecciones_usadas_altas:
                    continue

                det_bbox = det[:4]
                iou = calcular_iou(bbox_anterior, det_bbox)

                if iou > mejor_iou and iou > 0.3:  # Umbral IoU
                    mejor_iou = iou
                    mejor_idx = i

            if mejor_idx != -1:
                det = detecciones_altas[mejor_idx]
                det_bbox = det[:4]
                x, y, w, h = det_bbox
                centro = (x + w/2, y + h/2)

                self.tracks[track_id]['bbox'] = det_bbox
                self.tracks[track_id]['centro'] = centro
                self.tracks[track_id]['frames_perdidos'] = 0
                self.tracks[track_id]['conf'] = det[4]

                detecciones_usadas_altas.add(mejor_idx)
                tracks_asignados.add(track_id)
                tracks_actualizados.append((track_id, x, y, w, h))

        # === SEGUNDO MATCHING: Baja confianza para tracks no asignados ===
        tracks_no_asignados = set(self.tracks.keys()) - tracks_asignados
        detecciones_usadas_bajas = set()

        for track_id in tracks_no_asignados:
            if track_id not in self.tracks:
                continue

            mejor_iou = 0
            mejor_idx = -1
            bbox_anterior = self.tracks[track_id]['bbox']

            for i, det in enumerate(detecciones_bajas):
                if i in detecciones_usadas_bajas:
                    continue

                det_bbox = det[:4]
                iou = calcular_iou(bbox_anterior, det_bbox)

                if iou > mejor_iou and iou > 0.5:  # Umbral más alto para baja confianza
                    mejor_iou = iou
                    mejor_idx = i

            if mejor_idx != -1:
                det = detecciones_bajas[mejor_idx]
                det_bbox = det[:4]
                x, y, w, h = det_bbox
                centro = (x + w/2, y + h/2)

                self.tracks[track_id]['bbox'] = det_bbox
                self.tracks[track_id]['centro'] = centro
                self.tracks[track_id]['frames_perdidos'] = 0
                self.tracks[track_id]['conf'] = det[4]

                detecciones_usadas_bajas.add(mejor_idx)
                tracks_asignados.add(track_id)
                tracks_actualizados.append((track_id, x, y, w, h))

        # Incrementar frames perdidos para tracks no asignados
        for track_id in self.tracks.keys():
            if track_id not in tracks_asignados:
                self.tracks[track_id]['frames_perdidos'] += 1

        # Limpiar tracks perdidos
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['frames_perdidos'] > self.max_frames_perdidos:
                del self.tracks[track_id]

        # === CREAR NUEVOS TRACKS solo con alta confianza ===
        for i, det in enumerate(detecciones_altas):
            if i not in detecciones_usadas_altas:
                track_id = self.siguiente_id
                self.siguiente_id += 1

                det_bbox = det[:4]
                x, y, w, h = det_bbox
                centro = (x + w/2, y + h/2)

                self.tracks[track_id] = {
                    'bbox': det_bbox,
                    'centro': centro,
                    'frames_perdidos': 0,
                    'conf': det[4]
                }
                tracks_actualizados.append((track_id, x, y, w, h))

        return tracks_actualizados


class ROISelector:
    """Selector interactivo de ROI mediante polígono"""
    def __init__(self):
        self.puntos = []
        self.completado = False
        self.frame_display = None

    def click_evento(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.puntos.append((x, y))
            print(f"Punto {len(self.puntos)}: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.puntos) >= 3:
                self.completado = True
                print("ROI completado!")
            else:
                print("Necesitas al menos 3 puntos")

    def seleccionar_roi(self, frame):
        self.frame_display = frame.copy()
        window_name = 'Selecciona ROI - Click izquierdo: punto | Click derecho: terminar | ESC: cancelar'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.click_evento)

        print("\n" + "="*60)
        print("SELECCIÓN DE ROI")
        print("="*60)
        print("- Click IZQUIERDO: Agregar punto al polígono")
        print("- Click DERECHO: Finalizar polígono (mínimo 3 puntos)")
        print("- ESC: Cancelar y usar frame completo")
        print("="*60 + "\n")

        while True:
            display = self.frame_display.copy()

            for i, punto in enumerate(self.puntos):
                cv2.circle(display, punto, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (punto[0]+10, punto[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(self.puntos) > 1:
                for i in range(len(self.puntos) - 1):
                    cv2.line(display, self.puntos[i], self.puntos[i+1], (0, 255, 0), 2)
                if len(self.puntos) >= 3:
                    cv2.line(display, self.puntos[-1], self.puntos[0], (0, 255, 0), 1)

            cv2.putText(display, f"Puntos: {len(self.puntos)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Click IZQ: agregar | Click DER: terminar | ESC: cancelar",
                       (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Cancelado - usando frame completo")
                cv2.destroyAllWindows()
                return None

            if self.completado:
                break

        cv2.destroyAllWindows()

        if len(self.puntos) >= 3:
            roi_poly = np.array(self.puntos, dtype=np.int32)
            print(f"\nROI definido con {len(self.puntos)} puntos")
            return roi_poly

        return None


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


class AnalizadorCarrilesMejorado:
    def __init__(self, min_puntos_trayectoria=20, distancia_agrupacion=50,
                 min_trayectorias_por_carril=4):
        self.trayectorias_completas = []
        self.min_puntos_trayectoria = min_puntos_trayectoria
        self.distancia_agrupacion = distancia_agrupacion
        self.min_trayectorias_por_carril = min_trayectorias_por_carril

        self.carriles = []
        self.carriles_estables = []
        self.ultimo_calculo = 0
        self.frames_sin_cambio = 0

    def agregar_trayectoria(self, trayectoria):
        if len(trayectoria) >= self.min_puntos_trayectoria:
            trayectoria_array = np.array(list(trayectoria))
            self.trayectorias_completas.append(trayectoria_array)
            if len(self.trayectorias_completas) > 50:
                self.trayectorias_completas.pop(0)

    def calcular_similitud_trayectorias(self, traj1, traj2):
        if len(traj1) < 5 or len(traj2) < 5:
            return float('inf')

        indices1 = np.linspace(0, len(traj1) - 1, min(10, len(traj1))).astype(int)
        indices2 = np.linspace(0, len(traj2) - 1, min(10, len(traj2))).astype(int)

        puntos1 = traj1[indices1]
        puntos2 = traj2[indices2]

        distancias = cdist(puntos1, puntos2)
        return np.min(distancias, axis=1).mean()

    def agrupar_trayectorias(self):
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

            if len(grupo_actual) >= self.min_trayectorias_por_carril:
                grupos.append(grupo_actual)

        return grupos

    def calcular_linea_tendencia(self, trayectorias_grupo):
        todos_puntos = []
        for traj in trayectorias_grupo:
            todos_puntos.extend(traj)

        todos_puntos = np.array(todos_puntos)

        if len(todos_puntos) < 10:
            return None

        todos_puntos = todos_puntos[todos_puntos[:, 1].argsort()]

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

        try:
            puntos_promedio = puntos_promedio[puntos_promedio[:, 1].argsort()]
            k = min(3, len(puntos_promedio) - 1)
            spline = UnivariateSpline(puntos_promedio[:, 1], puntos_promedio[:, 0], k=k, s=1000)

            y_vals = np.linspace(puntos_promedio[:, 1].min(), puntos_promedio[:, 1].max(), 50)
            x_vals = spline(y_vals)

            linea_tendencia = np.column_stack([x_vals, y_vals])
            return linea_tendencia
        except:
            return puntos_promedio

    def validar_carriles_no_cruzan(self, carriles_candidatos):
        """Elimina carriles que se cruzan entre sí"""
        if len(carriles_candidatos) < 2:
            return carriles_candidatos

        carriles_validos = []

        for i, carril1 in enumerate(carriles_candidatos):
            linea1 = carril1['linea']
            es_valido = True

            for j, carril2 in enumerate(carriles_candidatos):
                if i == j:
                    continue

                linea2 = carril2['linea']

                if self.detectar_cruce(linea1, linea2):
                    # Si se cruzan, mantener el que tiene más trayectorias
                    if carril1['num_trayectorias'] < carril2['num_trayectorias']:
                        es_valido = False
                        break

            if es_valido:
                carriles_validos.append(carril1)

        return carriles_validos

    def detectar_cruce(self, linea1, linea2, umbral_cruce=30):
        """Detecta si dos líneas se cruzan"""
        if len(linea1) < 10 or len(linea2) < 10:
            return False

        # Muestrear puntos a lo largo de las líneas
        indices = np.linspace(0, min(len(linea1), len(linea2)) - 1, 5).astype(int)

        ordenes = []
        for idx in indices:
            if idx < len(linea1) and idx < len(linea2):
                x1 = linea1[idx][0]
                x2 = linea2[idx][0]
                ordenes.append(x1 < x2)

        if len(ordenes) >= 3:
            # Si cambia el orden, se cruzan
            cambios = sum(1 for i in range(len(ordenes)-1) if ordenes[i] != ordenes[i+1])
            return cambios > 0

        return False

    def validar_paralelismo(self, carriles_candidatos, max_desviacion=45):
        """Valida que los carriles sean paralelos (elimina diagonales extrañas)"""
        if len(carriles_candidatos) < 2:
            return carriles_candidatos

        carriles_validos = []

        for carril in carriles_candidatos:
            linea = carril['linea']

            if len(linea) >= 2:
                p1 = linea[0]
                p2 = linea[-1]

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angulo = np.degrees(np.arctan2(dy, dx))

                carril['angulo'] = angulo
                carriles_validos.append(carril)

        if len(carriles_validos) >= 2:
            angulos = [c['angulo'] for c in carriles_validos]
            angulo_medio = np.mean(angulos)

            carriles_paralelos = []
            for carril in carriles_validos:
                desviacion = abs(carril['angulo'] - angulo_medio)
                if desviacion < max_desviacion:
                    carriles_paralelos.append(carril)

            return carriles_paralelos

        return carriles_validos

    def fusionar_con_carriles_estables(self, carriles_nuevos):
        """Fusiona carriles nuevos con los estables para evitar saltos"""
        if not self.carriles_estables:
            return carriles_nuevos

        carriles_fusionados = []
        carriles_nuevos_usados = set()

        # Intentar emparejar carriles estables con nuevos
        for carril_estable in self.carriles_estables:
            x_estable = carril_estable['x_promedio']
            mejor_match = None
            mejor_distancia = float('inf')
            mejor_idx = -1

            for i, carril_nuevo in enumerate(carriles_nuevos):
                if i in carriles_nuevos_usados:
                    continue

                x_nuevo = carril_nuevo['x_promedio']
                distancia = abs(x_estable - x_nuevo)

                if distancia < 100 and distancia < mejor_distancia:
                    mejor_distancia = distancia
                    mejor_match = carril_nuevo
                    mejor_idx = i

            if mejor_match:
                # Fusionar con suavizado (70% estable, 30% nuevo)
                linea_estable = carril_estable['linea']
                linea_nueva = mejor_match['linea']

                if len(linea_estable) == len(linea_nueva):
                    linea_fusionada = 0.7 * linea_estable + 0.3 * linea_nueva
                else:
                    linea_fusionada = linea_nueva

                carril_fusionado = {
                    'linea': linea_fusionada,
                    'x_promedio': 0.7 * x_estable + 0.3 * mejor_match['x_promedio'],
                    'num_trayectorias': mejor_match['num_trayectorias'],
                    'estabilidad': carril_estable.get('estabilidad', 0) + 1
                }

                carriles_fusionados.append(carril_fusionado)
                carriles_nuevos_usados.add(mejor_idx)
            else:
                # Mantener carril estable si tiene alta estabilidad
                if carril_estable.get('estabilidad', 0) > 3:
                    carril_estable['estabilidad'] -= 1
                    carriles_fusionados.append(carril_estable)

        # Agregar carriles nuevos que no se emparejaron
        for i, carril_nuevo in enumerate(carriles_nuevos):
            if i not in carriles_nuevos_usados:
                carril_nuevo['estabilidad'] = 1
                if carril_nuevo['num_trayectorias'] >= self.min_trayectorias_por_carril:
                    carriles_fusionados.append(carril_nuevo)

        return carriles_fusionados

    def actualizar_carriles(self, frame_count):
        """Actualiza el cálculo de carriles cada 30 frames"""
        if frame_count - self.ultimo_calculo < 30:
            return False

        self.ultimo_calculo = frame_count

        grupos = self.agrupar_trayectorias()

        carriles_nuevos = []
        for grupo in grupos:
            linea = self.calcular_linea_tendencia(grupo)
            if linea is not None:
                x_promedio = np.mean(linea[:, 0])
                carriles_nuevos.append({
                    'linea': linea,
                    'x_promedio': x_promedio,
                    'num_trayectorias': len(grupo)
                })

        if not carriles_nuevos:
            return False

        # Aplicar validaciones
        carriles_sin_cruces = self.validar_carriles_no_cruzan(carriles_nuevos)
        carriles_paralelos = self.validar_paralelismo(carriles_sin_cruces)
        carriles_fusionados = self.fusionar_con_carriles_estables(carriles_paralelos)

        # Ordenar de izquierda a derecha
        carriles_fusionados.sort(key=lambda c: c['x_promedio'])

        self.carriles_estables = carriles_fusionados
        self.carriles = carriles_fusionados

        return len(self.carriles) > 0

    def asignar_carril(self, x, y):
        """Asigna un vehículo al carril más cercano"""
        if not self.carriles:
            return None

        mejor_carril = None
        mejor_distancia = float('inf')

        for i, carril in enumerate(self.carriles):
            linea = carril['linea']

            distancias = np.sqrt(np.sum((linea - np.array([x, y]))**2, axis=1))
            distancia_min = np.min(distancias)

            if distancia_min < mejor_distancia:
                mejor_distancia = distancia_min
                mejor_carril = i + 1

        if mejor_distancia < 80:
            return mejor_carril
        return None


def punto_en_poligono(punto, poligono):
    """Verifica si un punto está dentro de un polígono"""
    return cv2.pointPolygonTest(poligono, punto, False) >= 0


def main():
    parser = argparse.ArgumentParser(description='Sistema con ByteTrack (Sin Kalman)')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--modelo', type=str, default='yolov8n.pt')
    parser.add_argument('--min-puntos', type=int, default=20)
    parser.add_argument('--min-trayectorias', type=int, default=4)
    args = parser.parse_args()

    print("Cargando YOLO...")
    modelo = YOLO(args.modelo)

    video_path = args.video if args.video != '0' else 0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir {args.video}")
        return

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {ancho}x{alto}")

    ret, primer_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame")
        return

    selector = ROISelector()
    roi_poligono = selector.seleccionar_roi(primer_frame)

    if roi_poligono is not None:
        print(f"✓ ROI activo con {len(roi_poligono)} puntos")
        usar_roi = True
    else:
        print("✓ Procesando frame completo (sin ROI)")
        usar_roi = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"Mínimo trayectorias por carril: {args.min_trayectorias}")
    print("🎯 ByteTrack (Sin Kalman): ACTIVO")
    print("   - Alta confianza: >= 0.5")
    print("   - Baja confianza: 0.1 - 0.5")
    print("   - Max frames perdidos: 30")
    print("Procesando...")

    # ByteTrack sin Kalman
    tracker = ByteTrackSimple(high_thresh=0.5, low_thresh=0.1, max_frames_perdidos=30)
    mapa = MapaTrayectorias(ancho, alto, decay_rate=0.98)
    analizador = AnalizadorCarrilesMejorado(
        min_puntos_trayectoria=args.min_puntos,
        min_trayectorias_por_carril=args.min_trayectorias
    )

    frame_count = 0
    carriles_detectados = False

    colores = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
               (255, 255, 100), (255, 100, 255), (100, 255, 255)]
    colores_track = {}
    colores_carril = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                      (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    cv2.namedWindow('MODO APRENDIZAJE', cv2.WINDOW_NORMAL)
    cv2.namedWindow('CARRILES ACTIVOS', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detectar con YOLO - MENOR umbral para ByteTrack
        resultados = modelo(frame, verbose=False, conf=0.1)
        detecciones_con_conf = []

        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    conf = float(box.conf[0])

                    if usar_roi:
                        centro = (x + w//2, y + h//2)
                        if punto_en_poligono(centro, roi_poligono):
                            detecciones_con_conf.append([x, y, w, h, conf])
                    else:
                        detecciones_con_conf.append([x, y, w, h, conf])

        # Tracking con ByteTrack
        tracks_anteriores = set(tracker.tracks.keys())
        tracks = tracker.actualizar(detecciones_con_conf)
        tracks_actuales = set(tracker.tracks.keys())

        tracks_finalizados = tracks_anteriores - tracks_actuales
        for track_id in tracks_finalizados:
            if track_id in mapa.trayectorias:
                analizador.agregar_trayectoria(mapa.trayectorias[track_id])

        mapa.actualizar(tracks)

        if analizador.actualizar_carriles(frame_count):
            carriles_detectados = True

        carriles_activos = carriles_detectados and len(analizador.carriles) >= 2

        # ==================== VENTANA 1: MODO APRENDIZAJE ====================
        frame_aprendizaje = frame.copy()
        mapa_visual = mapa.obtener_mapa_visual()
        frame_aprendizaje = cv2.addWeighted(frame_aprendizaje, 0.7, mapa_visual, 0.3, 0)

        if usar_roi:
            cv2.polylines(frame_aprendizaje, [roi_poligono], True, (0, 255, 255), 2)

        # Dibujar trayectorias
        for track_id, trayectoria in mapa.trayectorias.items():
            if len(trayectoria) > 1:
                if track_id not in colores_track:
                    colores_track[track_id] = colores[len(colores_track) % len(colores)]
                color = colores_track[track_id]
                puntos = np.array(list(trayectoria), dtype=np.int32)
                cv2.polylines(frame_aprendizaje, [puntos], False, color, 2)

        # Dibujar vehículos
        for track_id, track_data in tracker.tracks.items():
            bbox = track_data['bbox']
            x, y, w, h = bbox
            conf = track_data.get('conf', 0)

            if track_id in colores_track:
                color = colores_track[track_id]
                cv2.rectangle(frame_aprendizaje, (int(x), int(y)),
                            (int(x+w), int(y+h)), color, 2)
                cv2.putText(frame_aprendizaje, f"ID:{track_id} ({conf:.2f})",
                           (int(x), int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(frame_aprendizaje, (10, 10), (350, 50), (0, 0, 0), -1)
        cv2.rectangle(frame_aprendizaje, (10, 10), (350, 50), (0, 255, 255), 2)
        cv2.putText(frame_aprendizaje, "MODO: APRENDIZAJE", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        roi_texto = "ROI: SI" if usar_roi else "ROI: NO"
        info = f"Frame: {frame_count} | Tracks: {len(tracker.tracks)} | Trayectorias: {len(analizador.trayectorias_completas)} | {roi_texto}"
        cv2.putText(frame_aprendizaje, info, (10, alto - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # ==================== VENTANA 2: CARRILES ACTIVOS ====================
        frame_carriles = frame.copy()

        if carriles_activos:
            mapa_visual_suave = mapa.obtener_mapa_visual()
            frame_carriles = cv2.addWeighted(frame_carriles, 0.95, mapa_visual_suave, 0.05, 0)

        if usar_roi:
            cv2.polylines(frame_carriles, [roi_poligono], True, (0, 255, 255), 2)

        # Dibujar líneas de carriles
        if carriles_activos:
            for i, carril in enumerate(analizador.carriles):
                linea = carril['linea'].astype(np.int32)
                color = colores_carril[i % len(colores_carril)]
                num_traj = carril['num_trayectorias']

                cv2.polylines(frame_carriles, [linea], False, color, 6)

                if len(linea) > 0:
                    punto_medio = linea[len(linea) // 2]
                    cv2.putText(frame_carriles, f"C{i+1} ({num_traj})",
                               (punto_medio[0] - 40, punto_medio[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Dibujar vehículos con asignación de carril
        for track_id, track_data in tracker.tracks.items():
            bbox = track_data['bbox']
            x, y, w, h = bbox
            centro_x = int(x + w/2)
            centro_y = int(y + h/2)

            if track_id in colores_track:
                color = colores_track[track_id]
                cv2.rectangle(frame_carriles, (int(x), int(y)),
                            (int(x+w), int(y+h)), color, 2)

                if carriles_activos:
                    carril_num = analizador.asignar_carril(centro_x, centro_y)
                    if carril_num:
                        cv2.putText(frame_carriles, f"ID:{track_id} C{carril_num}",
                                   (int(x), int(y) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        cv2.putText(frame_carriles, f"ID:{track_id}",
                                   (int(x), int(y) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame_carriles, f"ID:{track_id}",
                               (int(x), int(y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Indicador de modo
        if carriles_activos:
            cv2.rectangle(frame_carriles, (10, 10), (350, 50), (0, 0, 0), -1)
            cv2.rectangle(frame_carriles, (10, 10), (350, 50), (0, 255, 0), 2)
            cv2.putText(frame_carriles, "MODO: CARRILES ACTIVOS", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame_carriles, (10, 10), (450, 50), (0, 0, 0), -1)
            cv2.rectangle(frame_carriles, (10, 10), (450, 50), (255, 0, 0), 2)
            cv2.putText(frame_carriles, "ESPERANDO CARRILES...", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        info = f"Frame: {frame_count} | Tracks: {len(tracker.tracks)} | Carriles: {len(analizador.carriles)}"
        cv2.putText(frame_carriles, info, (10, alto - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('MODO APRENDIZAJE', frame_aprendizaje)
        cv2.imshow('CARRILES ACTIVOS', frame_carriles)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Completado")


if __name__ == "__main__":
    main()
