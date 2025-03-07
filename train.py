import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/11/yolo11_mobilenetV4.yaml')
    # # model.load('yolo11n-seg.pt') # loading pretrain weights
    # model.train(data='datasets/seg.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=100,
    #             batch=8,
    #             close_mosaic=0,
    #             workers=4,
    #             # device='0',
    #             optimizer='SGD',  # using SGD
    #             # patience=0, # close earlystop
    #             # resume=True, # 断点续训,YOLO初始化时选择last.pt
    #             # amp=False, # close amp
    #             # fraction=0.2,
    #             project='runs/train',
    #             name='mobilenetV4_N',
    #             )
    model=YOLO("runs/train/seg11_raw_Y/weights/best.pt")
    metrics=model.val()