from roboflow import Roboflow

rf = Roboflow(api_key="On4x587BnxR00EyYxk4W")
project = rf.workspace("karyotypezhongxin").project("autokary2022")
version  = project.version(1)

# use coco format — works for segmentation datasets
dataset = version.download("coco")

print("Download complete!")
print("Now run: python crop_chromosomes.py")




