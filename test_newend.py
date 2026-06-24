from ultralytics import YOLO

model = YOLO("Newend.pt")

model.predict(
    source="invoice_test2.png",
    save=True,
    project="results_Newend",
    name="newend",
    exist_ok=True,
    conf=0.25
)