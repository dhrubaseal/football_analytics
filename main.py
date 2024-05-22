from utils import read_video, save_video
from trackers import Tracker

def main():
    #Read Video
    video_frames = read_video(r'C:\Personal_Projects\projects\Computer Vision\football_analytics\data\raw\clips\0bfacc_4.mp4')

    #Initilize Tracker
    tracker = Tracker(r'C:\Personal_Projects\projects\Computer Vision\football_analytics\models\best.pt')

    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True,
                                       stub_path=r'C:\Personal_Projects\projects\Computer Vision\football_analytics\stubs\track_stubs.pkl')
    
    # Draw Output

    ## Draw Output Tracks

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #Save Video
    save_video(output_video_frames, r'C:\Personal_Projects\projects\Computer Vision\football_analytics\output_video\output_video.avi')

if __name__ == '__main__':
    main()