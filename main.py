from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    # Read Video
    video_frames = read_video(r'C:\Personal_Projects\projects\Computer Vision\football_analytics\data\clips\0bfacc_4.mp4')

    # Initilize Tracker
    tracker = Tracker(r'C:\Personal_Projects\projects\Computer Vision\football_analytics\models\best.pt')

    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True,
                                       stub_path=r'C:\Personal_Projects\projects\Computer Vision\football_analytics\stubs\track_stubs.pkl')
    
    # Interpolate Ball Positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
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