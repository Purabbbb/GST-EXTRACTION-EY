from ultralytics import YOLO

model = YOLO("gaurav.pt")

model.predict(
    source="invoice_test2.png",
    save=True,
    project="results_gaurav",
    name="gaurav",
    conf=0.25
)