# берем декоратор dataclass для удобного создания без init класса конфигурации и field для задания дефолтного пути сохранения
from dataclasses import dataclass, field
from pathlib import Path

# создаем с декоратором объект AppConfig с заданными параметрами, который будет использоваться во всем приложении для настройки его поведения.  обеспечивает централизованное место для управления конфигурацией приложения
@dataclass
class AppConfig:
    # Параметры  отображения окна камеры
    camera_width: int = 640
    camera_height: int = 360
    mirror_camera: bool = True # зеркальное отображение как у вебки

    # многопоточный захват камеры для повысшения производительности
    threaded_camera_capture: bool = True

    # Параметры детекции и предобработки кадров
    yolo_interval: int = 4 # интервал запуска детектора YOLO для обнаружения людей. Чем меньше значение, тем чаще будет запускаться детектор, что может улучшить точность трекинга, но увеличит нагрузку на CPU
    empty_scene_yolo_interval: int = 7 # если нет активных треков, ищем людей реже
    yolo_scale: float = 0.60 # сжатьие кадра перед подачей в детектор для повышения производительности. Сильное сжатие может снизить точность детекции, но ускорить обработку
    yolo_imgsz: int = 288 # размер входного изображения
    yolo_conf: float = 0.45 # уверенность детекции

    # Минимальный размер бокса, при котором объект еще обрабатываается
    min_box_w: int = 28
    min_box_h: int = 56
    min_identity_box_area: int = 5200 # минимальная площадь бокса для включения в identity-память, чтобы отсеивать слишком маленькие боксы, которые могут быть шумом и не содержать достаточно информации для надежного распознавания личности

    # Параметры жизненного цикла tracklet
    path_len: int = 80 # максимальная длина траектории, которая сохраняется для каждого tracklet. Длинные траектории могут занимать больше памяти, но обеспечивают более богатую историю движения объекта, что может помочь в распознавании личности и анализе поведения
    max_missed_detections: int = 8 # макс количесвто пропущенных деттекций при которых треклет все еще считается активным. Позволяет треклету пережить кратковременные occlusions или пропуски детекции
    min_confirmed_hits: int = 2 # минимальное количество подтвержденных попаданий (hits), необходимых, чтобы треклет считался подтвержденным. Это помогает отсеивать ложные треки, которые могут возникать из-за шумов в детекции

    # Сглаживание bbox после Kalman prediction и после реального измерения детектором
    prediction_center_alpha: float = 0.16 # насколько плавно двигать рамку при прогнозе Kalman
    prediction_size_alpha: float = 0.08

    measurement_center_alpha: float = 0.40 # насколько плавно двигать рамку при получении нового детектора
    measurement_size_alpha: float = 0.18
    
    size_jump_ratio: float = 0.18 # ограничение резких скачков размера рамки

    center_deadzone_px: int = 6 # размер зоны для центра рамки, в пределах которой не применяется сглаживание, чтобы избежать дрожания рамки при небольших движениях
    size_deadzone_px: int = 6

    # Пороги локального сопоставления detection <-> active tracklet и борьбы с дублями
    active_match_reid_threshold: float = 0.42 # насколько ReID-признак должен быть похож для локального сопоставления
    active_match_face_threshold: float = 0.38 # порог совпадения лица
    # фильтры дублей после YOLO, чтобы отсеивать сильно пересекающиеся боксы, которые могут принадлежать одному человеку
    detector_duplicate_iou: float = 0.58
    detector_nested_overlap: float = 0.76
    # фильтры дублей среди активных треков
    track_duplicate_iou: float = 0.64
    track_nested_overlap: float = 0.82

    # Параметры долговременной identity-памяти
    identity_min_hits: int = 4  # сколько наблюдений нужно, чтобы трек мог получить постоянный person_id
    identity_min_feature_updates: int = 2 # сколько обновлений признаков нужно, чтобы трек мог получить постоянный person_id. Это помогает убедиться, что у трека достаточно информации для надежного распознавания личности, прежде чем присваивать ему постоянный идентификатор
    identity_match_margin: float = 0.22
    #  пороги сходства для сопоставления человека с уже известной identity
    identity_reid_threshold: float = 0.70
    identity_color_threshold: float = 0.24
    identity_shape_threshold: float = 0.62
    identity_face_threshold: float = 0.36
    identity_max_age: int = 7200
    identity_bank_size: int = 18 # сколько образцов признаков хранить в памяти
    # как медленно обновлять портрет identity, чтобы не портить его шумными кадрами
    identity_feature_momentum: float = 0.95
    identity_color_momentum: float = 0.92
    identity_shape_momentum: float = 0.94

    # Параметры ReID-модели и color descriptor
    reid_backbone: str = "resnet50_gem" # тип модели признаков
    reid_weights: str = "reid_resnet50_msmt17.pth" # файл весов
    # размер кропа человека для ReID
    reid_input_height: int = 224
    reid_input_width: int = 112

    reid_padding: float = 0.08  # небольшой отступ вокруг bbox при вырезании человека
    reid_interval: int = 2 # интервал считывания признаков трека
    reid_force_count: int = 2
    max_reid_detections: int = 3 # макс кол-во детекций за проходку
    reid_num_parts: int = 3
    reid_gem_p: float = 3.0
    reid_global_weight: float = 1.0
    reid_part_weight: float = 0.65
    # детализация цветовой гистограммы
    reid_hist_h_bins: int = 12
    reid_hist_s_bins: int = 8

    # Параметры optional face backend
    # строки моделей лица
    face_detector_model: str = ""
    face_recognizer_model: str = ""
    # пороги детекции
    face_score_threshold: float = 0.88
    face_nms_threshold: float = 0.30

    # Параметры сохранения маршрутов и выходного видео.
    route_log_min_distance: int = 4
    writer_default_fps: float = 25.0
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "identity_tracking_output"
    )
