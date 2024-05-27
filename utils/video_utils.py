import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(video_frames, output_video_path):
    if not video_frames:
        print("Error: No video frames to save.")
        return
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_height, frame_width = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (frame_width, frame_height))

    for frame in video_frames:
        if frame is None:
            print("Warning: Skipping None frame.")
            continue
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_video_path}")

# def save_video(ouput_video_frames,output_video_path):
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
#     for frame in ouput_video_frames:
#         out.write(frame)
#     out.release()

