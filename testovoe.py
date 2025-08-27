import cv2
import numpy as np
import threading
import queue
import time

# фильтры 
def apply_filter(frame, mode):
    #применяем фильтр к каду в зависимости от режимап
    if mode == "denoise":
        # Сглаживание шума
        return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    elif mode == "contrast":

        # LAB: L - яркость, A и B - цвет
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    elif mode == "sharpen":

        blur = cv2.GaussianBlur(frame, (0, 0), 2.0)
        return cv2.addWeighted(frame, 1.8, blur, -0.8, 0)

    elif mode == "grayscale":

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return frame  

#потоки для захвата
def capture_frames(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret and not frame_queue.full():
            frame_queue.put(frame)

def process_frames(frame_queue, processed_queue, mode_getter, stop_event):
    #обработка кадров
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        processed = apply_filter(frame, mode_getter())
        if not processed_queue.full():
            processed_queue.put((frame, processed))


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("проблемы с камерой")
        return

    frame_queue = queue.Queue(maxsize=5)       # очередь для необработанных кадров
    processed_queue = queue.Queue(maxsize=5)   # очередь для обработанных кадров
    stop_event = threading.Event()             # остановка потока
    mode = {"value": "denoise"}                # выбранный фильтр


    t1 = threading.Thread(target=capture_frames, args=(cap, frame_queue, stop_event))
    t2 = threading.Thread(target=process_frames, args=(frame_queue, processed_queue, lambda: mode["value"], stop_event))
    t1.start()
    t2.start()

    print("1=denoise  2=contrast  3=sharpen  4=grayscale  Q=esc")

    # fps 
    prev_time = time.time()
    fps = 0
    count = 0

    while True:
        try:
            frame, processed = processed_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # рассчитываем FPS каждые 10 кадров
        count += 1
        if count >= 10:
            now = time.time()
            fps = count / (now - prev_time)
            prev_time = now
            count = 0

       
        cv2.putText(processed, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   
        combined = cv2.hconcat([frame, processed])
        cv2.imshow("Video (Q=quit, 1-4 фильтры)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # выход
            stop_event.set()
            break
        elif key == ord("1"):
            mode["value"] = "denoise"
        elif key == ord("2"):
            mode["value"] = "contrast"
        elif key == ord("3"):
            mode["value"] = "sharpen"
        elif key == ord("4"):
            mode["value"] = "grayscale"

    t1.join()
    t2.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
