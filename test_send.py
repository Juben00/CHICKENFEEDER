import requests

url = "http://127.0.0.1:5000/api/count_pellets"
image_path = "./test.jpg"

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print(response.json())