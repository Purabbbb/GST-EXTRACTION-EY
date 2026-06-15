from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="invoice_test2.png",
    save=True,
    conf=0.25
)