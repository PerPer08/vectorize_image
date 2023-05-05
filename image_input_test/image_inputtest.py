from PIL import Image

filename = input("Enter the path to your image file: ")
try:
    with Image.open(filename) as im:
        # Perform operations on the image here
        # For example, you could display the image
        im.show()
except IOError:
    print("Could not open image file")