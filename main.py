from utils import read_video, save_video
from trackers import Tracker
import cv2

def main():
    # Read Video
    video_frames = read_video(r'C:\Personal_Projects\projects\Computer Vision\football_analytics\data\clips\0bfacc_4.mp4')

    # Initilize Tracker
    tracker = Tracker(r'C:\Personal_Projects\projects\Computer Vision\football_analytics\models\best.pt')

    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True,
                                       stub_path=r'C:\Personal_Projects\projects\Computer Vision\football_analytics\stubs\track_stubs.pkl')
    
    # # Save croppped image of player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     # Crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # Save the cropped image
    #     cv2.imwrite(f'output_image/cropped_img.jpg', cropped_image)
        
    #     break


    # Draw Output

    ## Draw Output Tracks

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, r'C:\Personal_Projects\projects\Computer Vision\football_analytics\output_video\output_video.avi')

if __name__ == '__main__':
    main()