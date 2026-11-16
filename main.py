from building_recognition import recognize_building
from location import get_image_coordinates

if __name__ == "__main__":
    image_path = "kepek/test3.jpg"
    coords = get_image_coordinates(image_path)
    if coords:
        lat, lon = coords
        print(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")
    else:
        print("No GPS coordinates found in the image.")

    result_json = recognize_building(image_path, coords=coords)
    print(result_json)
