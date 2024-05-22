from ultralytics import YOLO

model = YOLO(r'C:\Personal_Projects\projects\Computer Vision\football_analytics\models\best.pt')

results = model.predict(r'C:\Personal_Projects\projects\Computer Vision\football_analytics\data\raw\clips\0a2d9b_0.mp4', save=True)

print(results[0])

print('-------------------------------')

for box in results[0].boxes:
    print(box)